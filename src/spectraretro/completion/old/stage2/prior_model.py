"""Isolated, rotation-invariant chemical residual adapter for official EDM."""
from contextlib import contextmanager
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

import bootstrap  # noqa: F401
from src.lightning import DDPM
from src.datasets import get_dataloader
from prior_data import PriorDataset, prior_collate
from prior_features import FEATURE_DIM, feature_contract


class EpochRandomSampler(Sampler):
    """Epoch-indexed permutations avoid resume-dependent DataLoader RNG draws."""
    def __init__(self, dataset, seed, model):
        self.dataset, self.seed, self.model = dataset, seed, model

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + int(self.model.current_epoch))
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)


class ChemicalResidual(nn.Module):
    def __init__(self, hidden_nf, prior_hidden_nf=64, scale=0.1):
        super().__init__()
        self.scale = float(scale)
        self.encoder = nn.Sequential(nn.Linear(2 * FEATURE_DIM, prior_hidden_nf), nn.SiLU(),
                                     nn.Linear(prior_hidden_nf, prior_hidden_nf), nn.SiLU())
        self.gate = nn.Sequential(nn.Linear(hidden_nf + prior_hidden_nf + 1, prior_hidden_nf),
                                  nn.SiLU(), nn.Linear(prior_hidden_nf, 1), nn.Sigmoid())
        self.projection = nn.Linear(prior_hidden_nf, hidden_nf)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden, features, time, node_mask):
        prior = self.encoder(features)
        gate = self.gate(torch.cat([hidden, prior, time], dim=-1))
        residual = self.scale * gate * torch.tanh(self.projection(prior))
        return hidden + residual * node_mask


class PriorDynamics(nn.Module):
    """Keep vendor dynamics intact and inject only at its scalar embedding."""
    def __init__(self, base, enabled=True, prior_hidden_nf=64, scale=0.1):
        super().__init__()
        if base.model != "egnn_dynamics":
            raise ValueError("The scalar chemical adapter is defined for the existing EGNN")
        self.base = base
        self.enabled = enabled
        # Preserve base initialization and subsequent data/noise RNG, irrespective
        # of whether an adapter is enabled. CPU constructor does not touch CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            self.adapter = ChemicalResidual(base.dynamics.hidden_nf, prior_hidden_nf, scale)
        self._condition = None
        self._step = None
        self.base.dynamics.embedding.register_forward_hook(self._embedding_hook)

    @contextmanager
    def condition(self, data):
        previous = self._condition
        if self.enabled:
            if "chemistry_prior" not in data:
                raise ValueError("Prior model requires observed-graph features")
            # Only observed fragment nodes can contribute; unknown target nodes
            # are prohibited even if a caller accidentally supplies them.
            features = data["chemistry_prior"]
            fragment = data["fragment_mask"]
            if features.shape[-1] != FEATURE_DIM:
                raise ValueError("Wrong chemical feature schema")
            if torch.any(features * (1 - fragment)):
                raise ValueError("Target/linker nodes may not contain chemical prior features")
            self._condition = (features, fragment, data["anchors"])
        try:
            yield
        finally:
            self._condition = previous

    def _embedding_hook(self, module, inputs, hidden):
        if not self.enabled:
            return hidden
        if self._step is None:
            raise RuntimeError("Chemical adapter called outside a scoped denoising step")
        features, time, mask = self._step
        return self.adapter(hidden, features, time, mask)

    def forward(self, t, xh, node_mask, linker_mask, edge_mask, context):
        if not self.enabled:
            return self.base.forward(t, xh, node_mask, linker_mask, edge_mask, context)
        if self._condition is None:
            raise RuntimeError("Prior dynamics require an observed-synthon condition")
        original, fragment, anchors = self._condition
        batch_size, n_nodes = xh.shape[:2]
        if original.shape[0] != batch_size:
            raise ValueError("Condition batch does not match denoising batch")
        features = xh.new_zeros(batch_size, n_nodes, FEATURE_DIM)
        for index in range(batch_size):
            count = int(fragment[index].sum().item())
            if count > n_nodes:
                raise ValueError("Denoising template has fewer atoms than its observed synthon")
            features[index, :count] = original[index, :count].to(xh)
            if not torch.equal(context[index, :count, -1], torch.ones_like(context[index, :count, -1])):
                raise ValueError("Known-fragment prefix was not retained in the denoising template")
        anchors = anchors.to(xh)
        pooled = (original.to(xh) * anchors).sum(1) / anchors.sum(1).clamp_min(1)
        features = torch.cat([features, pooled[:, None, :].expand(-1, n_nodes, -1)], dim=-1)
        if t.numel() == 1:
            time = t.reshape(1, 1, 1).expand(batch_size, n_nodes, 1)
        else:
            time = t.reshape(batch_size, 1, 1).expand(-1, n_nodes, -1)
        previous = self._step
        available = (original.abs().sum(dim=(1, 2)) > 0).to(xh).reshape(batch_size, 1, 1)
        self._step = (features.reshape(batch_size * n_nodes, -1),
                      time.reshape(-1, 1), (node_mask * available).reshape(-1, 1))
        try:
            return self.base.forward(t, xh, node_mask, linker_mask, edge_mask, context)
        finally:
            self._step = previous


class PriorDDPM(DDPM):
    def __init__(self, prior_enabled=True, prior_hidden_nf=64, prior_scale=0.1, **kwargs):
        if kwargs.get("diffusion_process", "edm") != "edm":
            raise ValueError("This experiment uses unchanged conventional EDM")
        if kwargs.get("n_dims") != 3 or kwargs.get("inpainting", False):
            raise ValueError("This experiment is fixed-synthon 3D completion")
        super().__init__(**kwargs)
        self.save_hyperparameters({"prior_enabled": prior_enabled, "prior_hidden_nf": prior_hidden_nf,
                                   "prior_scale": prior_scale})
        self.edm.dynamics = PriorDynamics(self.edm.dynamics, prior_enabled, prior_hidden_nf, prior_scale)
        self.train_generator = torch.Generator().manual_seed(torch.initial_seed())
        self.val_generator = torch.Generator().manual_seed(torch.initial_seed() + 1)
        self.shuffle_seed = torch.initial_seed()

    def _checkpoint_contract(self):
        result = super()._checkpoint_contract()
        result.update({"protocol": "isolated_mechanism_prior_edm_20260919_v1",
                       "chemical_prior_enabled": self.hparams.prior_enabled,
                       "chemical_prior": feature_contract()})
        return result

    def setup(self, stage=None):
        if stage in ("fit", None):
            if self.train_dataset is not None and self.val_dataset is not None:
                return
            self.train_dataset = PriorDataset(self.data_path, self.train_data_prefix, "cpu")
            self.val_dataset = PriorDataset(self.data_path, self.val_data_prefix, "cpu")
        elif stage in ("val", "validate"):
            self.val_dataset = PriorDataset(self.data_path, self.val_data_prefix, "cpu")
        else:
            raise NotImplementedError(stage)

    def train_dataloader(self, collate_fn=prior_collate):
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=collate_fn,
                          sampler=EpochRandomSampler(self.train_dataset, self.shuffle_seed, self),
                          generator=self.train_generator)

    def val_dataloader(self, collate_fn=prior_collate):
        return DataLoader(self.val_dataset, self.batch_size, collate_fn=collate_fn,
                          generator=self.val_generator)

    def forward(self, data, training):
        with self.edm.dynamics.condition(data):
            return super().forward(data, training)

    def sample_chain(self, data, sample_fn=None, keep_frames=None):
        with self.edm.dynamics.condition(data):
            return super().sample_chain(data, sample_fn, keep_frames)
