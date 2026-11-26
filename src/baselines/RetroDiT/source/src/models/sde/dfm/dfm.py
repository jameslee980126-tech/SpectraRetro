import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from torch import Tensor
from typing import Literal
from omegaconf import DictConfig

from .dfm_utils import *
from .time_distorter import TimeDistorter
from .rate_matrix import RateMatrixDesigner
from src.data.batch_class import DenseGraph
from ..sde import SDE
from src.data.batch_class import *


class DFM(SDE):
    def __init__(
            self,
            from_var: list[str], to_var: list[str],
            conditional: bool,
            model_transition: Literal['predefined', 'uniform', 'absorbing'],
            time_distortion: Literal[
                'identity', 'cosine', 'revcos', 'polyinc', 'polydec'
            ],
            train: DictConfig, sample: DictConfig,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.conditional = conditional

        self.from_var = from_var
        self.to_var = to_var
        assert len(self.from_var) == len(self.to_var)
        self.lambda_train_x = train.lambda_train_x
        self.lambda_train_e = train.lambda_train_e
        self.entropy_weight_x = train.entropy_weight_x
        self.entropy_weight_e = train.entropy_weight_e
        self.p_loss_only = train.p_loss_only

        self.time_distortion = time_distortion

        # define limit distribution
        self.model_transition = model_transition

        # time distortor for both training and sampling steps
        self.time_distorter = TimeDistorter(
            alpha=sample.alpha,
            beta=sample.beta,
        )

        # rate matrix designer
        self.rate_matrix_designer = RateMatrixDesigner(
            rdb=sample.rdb,
            rdb_crit=sample.rdb_crit,
            eta=sample.eta,
            omega=sample.omega,
            model_transition=self.model_transition,
        )
        self.loss_type = train.loss_type
        assert self.loss_type in ['ce', 'vlb'], f"Unsupported loss type: {self.loss_type}"
        
        self.beam_size = sample.get('beam_size', 1)
        self.beam_temperature = sample.get('beam_temperature', 1.0)  # Temperature for beam search diversity
        self.beam_strategy = sample.get('beam_strategy', 'temperature')  # 'temperature', 'stratified', 'topk'
        self.confidence = []
        
        
    def get_x0_x1(self, x0, x1):
        device = x0.device
        if self.model_transition == "uniform":
            limit_x0 = torch.ones_like(x0) / x0.size(-1)
            target_x1 = x1
        elif self.model_transition == "absorbing":
            new_shape = list(x0.shape)
            new_shape[-1] += 1
            limit_x0 = torch.zeros(new_shape, device=device)
            limit_x0[..., -1] = 1
            target_x1 = torch.zeros(new_shape, device=device)
            target_x1[..., :-1] = x1
        elif self.model_transition == "predefined":
            limit_x0 = x0
            target_x1 = x1
        else:
            raise ValueError(f"Unknown transition model: {self.model_transition}")
        return limit_x0, target_x1
        
    def discrete_forward(
            self,
            t: Tensor,
            x0: Tensor = None,
            x1: Tensor = None,
            node_mask: Tensor | None = None
        ) -> dict[str, Tensor]:
        """
        Args:
            t: [B, ]
            x: [B, ..., K], already one-hot
        """
        """Sample noise and apply it to the data."""

        assert x0.shape == x1.shape, "x0 and x1 must have the same shape."    
        assert t.ndim == 1

        t_float = self.time_distorter.sample_ft(t, self.time_distortion)
        prob_xt = p_xt_g_x01(
            x0=x0, x1=x1, t=t_float
        )
        if x0.ndim == 3:
            sampled_t = sample_discrete_node_features(
                probX=prob_xt, node_mask=node_mask
            )
        elif x0.ndim == 4:
            sampled_t = sample_discrete_edge_features(
                probE=prob_xt, node_mask=node_mask
            )
        else:
            raise ValueError(f"Unsupported x0 shape: {x0.shape}")
        xt = F.one_hot(sampled_t, num_classes=x0.size(-1))
        return xt.float()

    def __call__(
            self, batch: GraphBatch, t: Tensor | None = None
        ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        if t is None:
            t = torch.rand(
                size=(batch.p_X.size(0), ),
                device=batch.p_X.device
            )
        model_cond = {k: batch[k] for k in self.cond_on}
        
        noise_data = {}
        for from_var, to_var in zip(self.from_var, self.to_var):
            # Get transformed x0 and x1 based on model transition type
            x0_transformed, x1_transformed = self.get_x0_x1(batch[from_var], batch[to_var])
            
            # Apply discrete forward process
            noise_data['t' + from_var[1:]] = self.discrete_forward(
                t=t, 
                x0=x0_transformed, 
                x1=x1_transformed, 
                node_mask=batch.node_mask
            )
        model_input = {**noise_data, **model_cond}
        target = {k: batch[k] for k in self.to_var}
        target['p_mask'] = getattr(batch, 'p_mask', None)
        return t, model_input, target

    def discrete_loss(
            self,
            t: Tensor, pred: Tensor, target: Tensor,
            loss_mask: Tensor, p_mask: Tensor
        ) -> Tensor:

        if pred.ndim == 3:
            if self.p_loss_only:
                loss_mask = ~p_mask
            logging_suffix = 'X'
            sample_size = loss_mask.sum()
            
            pred_flat = pred.view(-1, pred.size(-1))  # [B*N, K]
            target_flat = target.view(-1, target.size(-1))  # [B*N, K]
            mask_flat = loss_mask.reshape(-1)  # [B*N]
            
            # Different losses:
            if self.loss_type == 'ce':
                loss = F.cross_entropy(pred_flat, target_flat.argmax(dim=-1), reduction='none')
            elif self.loss_type == 'vlb':
                normalized_preds = F.log_softmax(pred_flat, dim=-1)
                normalized_target = target_flat + 1e-10
                normalized_target = normalized_target / torch.sum(normalized_target, dim=-1, keepdim=True)
                loss = F.kl_div(normalized_preds, normalized_target, reduction='none').sum(dim=-1)

            loss = (loss * mask_flat.float()).sum() / mask_flat.sum().clamp(min=1) * self.lambda_train_x
            
            if self.entropy_weight_x > 0:
                raise NotImplementedError('not implemented p_mask')
            
        elif pred.ndim == 4:
            logging_suffix = 'E'
            edge_mask = loss_mask.unsqueeze(-1) * loss_mask.unsqueeze(-2)  # [B, N, N]
            sample_size = edge_mask.sum()
            
            pred_flat = pred.view(-1, pred.size(-1))  # [B*N*N, K]
            target_flat = target.view(-1, target.size(-1))  # [B*N*N, K]
            mask_flat = edge_mask.view(-1)  # [B*N*N]
            
            # Different losses:
            if self.loss_type == 'ce':
                loss = F.cross_entropy(pred_flat, target_flat.argmax(dim=-1), reduction='none')
            elif self.loss_type == 'vlb':
                normalized_preds = F.log_softmax(pred_flat, dim=-1)
                normalized_target = target_flat + 1e-10
                normalized_target = normalized_target / torch.sum(normalized_target, dim=-1, keepdim=True)
                loss = F.kl_div(normalized_preds, normalized_target, reduction='none').sum(dim=-1)

            loss = (loss * mask_flat.float()).sum() / mask_flat.sum().clamp(min=1) * self.lambda_train_e

            if self.entropy_weight_e > 0:
                # Compute entropy regularization for edges with normalization
                log_probs = F.log_softmax(pred_flat, dim=-1)  # [B*N*N, K] - numerically stable
                probs = F.softmax(pred_flat, dim=-1)  # [B*N*N, K]
                entropy = -torch.sum(probs * log_probs, dim=-1)  # [B*N*N]
                
                # Normalize entropy by maximum possible entropy (log(K))
                max_entropy = torch.log(torch.tensor(pred.size(-1), dtype=entropy.dtype, device=entropy.device))
                normalized_entropy = entropy / max_entropy  # [B*N*N] in range [0, 1]
                
                entropy_reg = -(normalized_entropy * mask_flat.float()).sum() / mask_flat.sum().clamp(min=1)
                loss += self.entropy_weight_e * entropy_reg

        else:
            raise ValueError()
        
        pred_labels = torch.argmax(pred, dim=-1)
        target_labels = torch.argmax(target, dim=-1)
        accuracy = (pred_labels == target_labels)

        if pred.ndim == 3:
            valid_accuracy = (accuracy * loss_mask).sum(dim=-1)
            valid_count = loss_mask.sum(dim=-1).clamp(min=1)
            accuracy = (valid_accuracy / valid_count).mean()
        else:
            accuracy = (accuracy * edge_mask).sum() / edge_mask.sum().clamp(min=1)

        logging_output = {
            f"loss_{logging_suffix}": loss.item(),
            f"acc_{logging_suffix}": accuracy.item(),
            f"bsz_{logging_suffix}": sample_size.item(),
        }
        
        # Add entropy regularization to logging if applicable
        entropy_weight = self.entropy_weight_x if pred.ndim == 3 else self.entropy_weight_e
        if entropy_weight > 0:
            raise NotImplementedError()

        return loss, logging_output
        
    def loss(
            self,
            t: Tensor,
            node_mask: Tensor,
            pred: dict[str, Tensor],
            target: dict[str, Tensor],
            **kwargs
        ):
        loss = 0
        logging_output = {}

    
        for model_k, target_k in zip(
            ['X', 'E'], self.to_var
        ):
            loss_result = self.discrete_loss(
                t, pred[model_k], target[target_k],
                node_mask, target['p_mask']
            )
            loss += loss_result[0]
            logging_output = logging_output | loss_result[1]

        return loss, logging_output

    @torch.inference_mode()
    def sampling(
        self,
        net: nn.Module,
        batch: GraphSamplingBatch,
        classifier: nn.Module = None,
        is_val: bool = False
    ) -> GraphBatch:
        """
        Sampling function for discrete flow networks.
        
        Args:
            net: Neural network model
            batch: Input batch data
            is_val: 
        
        Returns:
            DataBatch: Generated samples
        """
        if classifier is not None:
            assert batch.target is not None
            target = torch.argmax(batch.target, -1) # [B, k] 

        batch_size = batch.node_mask.shape[0]
        device = batch.node_mask.device
        # Fix: Correct the logic for sampling_steps
        num_steps = self.val_sampling_steps if is_val else self.test_sampling_steps
        
        # Fix: Use correct dimensions for initialization
        transformed_node_x0, _ = self.get_x0_x1(batch.p_X, torch.zeros_like(batch.p_X))
        transformed_edge_x0, _ = self.get_x0_x1(batch.p_E, torch.zeros_like(batch.p_E))
        if self.model_transition != 'predefined':
            z_0 = sample_discrete_feature_noise(transformed_node_x0, transformed_edge_x0, batch.node_mask)
            transformed_node_x0, transformed_edge_x0 = z_0.X, z_0.E
        
        # Initialize denoising batch
        current_batch = batch.to_data(
            r_X=transformed_node_x0,
            r_E=transformed_edge_x0,
        )
        # masking
        current_batch.update_r_(
            r_X=transformed_node_x0,
            r_E=transformed_edge_x0
        )
        
        fixed_nodes = (batch.p_X[..., -1] == 0).unsqueeze(-1)
        modifiable_nodes = (batch.p_X[..., -1] == 1).unsqueeze(-1)
        assert torch.all(fixed_nodes | modifiable_nodes)
        X_0, E_0 = current_batch.r_X, current_batch.r_E

        cond_current = current_batch.cond
        
        # Sanity check for edge symmetry
        assert (E_0 == torch.transpose(E_0, -3, -2)).all(), "Edges must be symmetric"
        
        nll = torch.zeros(batch_size, device=device, dtype=torch.float64)
        ell = torch.zeros(batch_size, device=device, dtype=torch.float64)
         
        iterator = tqdm(range(num_steps), leave=False, dynamic_ncols=True)
        
        # self.confidence.append({})
            
        # Main sampling loop
        for step in iterator:
            # self.confidence[-1][step] = []
            # Calculate normalized time steps
            t_current = step / (num_steps)
            t_next = (step + 1) / (num_steps)
            
            if ("absorb" in self.model_transition) and (step == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_current = t_current + 1e-6
            
            # Create time tensors
            t_tensor = torch.full((batch_size, 1), t_current, device=device, dtype=cond_current.dtype)
            s_tensor = torch.full((batch_size, 1), t_next, device=device, dtype=cond_current.dtype)

            # Apply time distortion if configured
            t_distorted = self.time_distorter.sample_ft(t_tensor, self.time_distortion)
            s_distorted = self.time_distorter.sample_ft(s_tensor, self.time_distortion)
            
            # Get network predictions
            model_cond = {k: current_batch[k] for k in self.cond_on}
            noise_data = {
                't_X': current_batch.r_X,
                't_E': current_batch.r_E
            }
            model_input = {**noise_data, **model_cond}
            
            pred = net(
                t=t_distorted.squeeze(), node_mask=current_batch.node_mask, **model_input
            )
            
            # Sample next state
            next_state, node_log_likelihood, edge_log_likelihood = self.sample_p_zs_given_zt(
                pred=pred,
                t=t_distorted,
                s=s_distorted,
                current_batch=current_batch,
                X_0=X_0,
                E_0=E_0,
                cond=cond_current,
                node_mask=batch.node_mask,
            )

            # Update current state
            nll += node_log_likelihood
            ell += edge_log_likelihood
            
            current_batch.update_r_(
                r_X=next_state.X,
                r_E=next_state.E
            )

            cond_current = next_state.y
            
        t_tensor = torch.full((batch_size, 1), 1, device=device, dtype=cond_current.dtype)
        t_distorted = self.time_distorter.sample_ft(t_tensor, self.time_distortion)
        model_cond = {k: current_batch[k] for k in self.cond_on}
        noise_data = {
            't_X': current_batch.r_X,
            't_E': current_batch.r_E
        }
        model_input = {**noise_data, **model_cond}
        
        pred = net(
            t=t_distorted.squeeze(),
            node_mask=current_batch.node_mask,
            **model_input
        )
        # Ignore virtual classes if applicable
        if self.model_transition == "absorbing":
            current_batch.update_r_(
                r_X=F.softmax(pred['X'][..., :-1], dim=-1),
                r_E=F.softmax(pred['E'][..., :-1], dim=-1)
            )
        else:
            current_batch.update_r_(
                r_X=F.softmax(pred['X'], dim=-1),
                r_E=F.softmax(pred['E'], dim=-1)
            )

        return current_batch

    def sample_p_zs_given_zt(
        self,
        pred: dict[str, Tensor],
        t: Tensor,
        s: Tensor, 
        current_batch: GraphBatch,
        X_0: Tensor,
        E_0: Tensor,
        cond: Tensor,
        node_mask: Tensor,
    ) -> DenseGraph:
        """
        Sample from zs ~ p(zs | zt) during the denoising process.
        
        Args:
            pred: Model predictions containing 'X' and 'E' keys
            t, s: Current and next time steps
            X_t, E_t, y_t: Current state
            X_0, E_0: initial state
            cond: Conditional data  
            node_mask: Mask for valid nodes
            
        Returns:
            PlaceHolder: Next state samples
        """
        device = X_0.device
        dt = (s - t)[0]
        X_t, E_t, y_t = current_batch.r_X, current_batch.r_E, current_batch.cond

        # Normalize predictions to probabilities
        pred_X = F.softmax(pred['X'], dim=-1)  # [B, N, D_x]
        pred_E = F.softmax(pred['E'], dim=-1)  # [B, N, N, D_e]

        # Prepare graph representations
        G_pred = (pred_X, pred_E)
        G_current = (X_t, E_t)
        G_0 = (X_0, E_0)

        # Compute rate matrices
        R_t_X, R_t_E = self.rate_matrix_designer.compute_graph_rate_matrix(
            t=t,
            node_mask=node_mask,
            G_t=G_current,
            G_0=G_0,
            G_1_pred=G_pred,
        )

        prob_X, prob_E = self.compute_step_probs(
            R_t_X=R_t_X, 
            R_t_E=R_t_E, 
            X_t=X_t, 
            E_t=E_t, 
            dt=dt
        )

        if s[0] == 1.0:
            prob_X, prob_E = pred_X, pred_E

        # Sample discrete features
        sampled_next = sample_discrete_features(
            probX=prob_X, 
            probE=prob_E, 
            node_mask=node_mask
        )

        # Convert to one-hot encoding
        X_s = F.one_hot(sampled_next.X, num_classes=X_0.shape[-1]).float()
        E_s = F.one_hot(sampled_next.E, num_classes=E_0.shape[-1]).float()
        

        # Sanity checks
        assert (E_s == torch.transpose(E_s, 1, 2)).all(), "Generated edges must be symmetric"
        assert X_t.shape == X_s.shape, f"Shape mismatch: X_t {X_t.shape} vs X_s {X_s.shape}"
        assert E_t.shape == E_s.shape, f"Shape mismatch: E_t {E_t.shape} vs E_s {E_s.shape}"

        fixed_nodes = (current_batch.p_X[..., -1] == 0).unsqueeze(-1)
        modifiable_nodes = (current_batch.p_X[..., -1] == 1).unsqueeze(-1)
        assert torch.all(fixed_nodes | modifiable_nodes)
        
        # Compute log-likelihood of the sampled states
        # We calculate P(X_s | X_t, t) where X_s is the sampled next state
        eps = 1e-10
        
        # Node log-likelihood: log P(X_s | X_t, t)
        node_log_likelihood = torch.log(prob_X + eps)  # [B, N, D_x]
        node_log_likelihood = (node_log_likelihood * X_s).sum(-1)  # [B, N] - sum over states
        node_log_likelihood = (node_log_likelihood * node_mask).sum(-1)  # [B] - sum over nodes with masking

        # Edge log-likelihood: log P(E_s | E_t, t)  
        edge_log_likelihood = torch.log(prob_E + eps)  # [B, N, N, D_e]
        edge_log_likelihood = (edge_log_likelihood * E_s).sum(-1)  # [B, N, N] - sum over edge states
        
        # Apply edge masking: only consider edges between valid nodes
        edge_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)  # [B, N, N]
        edge_log_likelihood = (edge_log_likelihood * edge_mask).sum(-1).sum(-1)  # [B] - sum over edges with masking

        # Handle y dimension based on conditional setting
        if self.conditional:
            y_output = y_t
        else:
            y_output = torch.zeros([y_t.shape[0], 0], device=device)

        # Create output placeholder and apply masking
        output = DenseGraph(X=X_s, E=E_s, y=y_output)
        output = output.mask(node_mask).type_as(y_t)
        
        return output, node_log_likelihood, edge_log_likelihood

    def compute_step_probs(
        self, 
        R_t_X: Tensor, 
        R_t_E: Tensor, 
        X_t: Tensor, 
        E_t: Tensor, 
        dt: float
    ) -> tuple[Tensor, Tensor]:
        """
        Compute step transition probabilities for discrete features using matrix exponential.
        
        Args:
            R_t_X, R_t_E: Rate matrices for nodes and edges
            X_t, E_t: Current one-hot states
            dt: Time step size
            
        Returns:
            Tuple of transition probabilities for nodes and edges
        """
        # Compute off-diagonal transition probabilities
        step_probs_X = R_t_X * dt  # [B, N, D_x]
        step_probs_E = R_t_E * dt  # [B, N, N, D_e]

        # Get current state indices
        current_X_idx = X_t.argmax(dim=-1)  # [B, N]
        current_E_idx = E_t.argmax(dim=-1)  # [B, N, N]

        # Zero out diagonal entries (transitions to current state)
        step_probs_X.scatter_(-1, current_X_idx.unsqueeze(-1), 0.0)
        step_probs_E.scatter_(-1, current_E_idx.unsqueeze(-1), 0.0)

        # Set diagonal entries to ensure probabilities sum to 1
        # P(stay in current state) = 1 - P(transition to other states)
        stay_prob_X = (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        stay_prob_E = (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        
        step_probs_X.scatter_(-1, current_X_idx.unsqueeze(-1), stay_prob_X)
        step_probs_E.scatter_(-1, current_E_idx.unsqueeze(-1), stay_prob_E)

        return step_probs_X.clone(), step_probs_E.clone()
