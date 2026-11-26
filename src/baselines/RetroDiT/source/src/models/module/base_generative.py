import os
import json
import time
import shutil

from rdkit import RDLogger
from functools import partial
from hydra.core.hydra_config import HydraConfig

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.aggregation import BaseAggregator

from lightning.pytorch.utilities import grad_norm
from lightning import LightningModule, seed_everything

from ..sde import SDE
from src.data.dataset import load_graph_vocab
from src.utils import RankedLogger
from src.utils.sampling_store import ResultStore

__all__ = ['BaseGenerativeModule']

log = RankedLogger(__name__, rank_zero_only=True)
RDLogger.DisableLog('rdApp.*')

class BaseGenerativeModule(LightningModule):
    def __init__(
        self,
        sde: SDE,
        net: nn.Module,

        graph_vocab: str, 
        optimizer, scheduler,
        sample_store: partial[ResultStore],
        val_top_ks: list[int] = [1, 3],
        test_top_ks: list[int] = [1, 3, 5, 10],
        sampling_seed: int | None = 42,
        ckpt_path = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['sde', 'net'], logger=False)
        
        self.sde = sde
        self.net = net
        _, self.x_dec, _, self.e_dec = load_graph_vocab(graph_vocab)

        if optimizer:
            assert 5 in val_top_ks, "val_top_ks must contain 5 for retrosynthesis metrics"
        self.val_top_ks  = val_top_ks
        self.test_top_ks = test_top_ks
        self.sampling_seed = 42 if sampling_seed is None else sampling_seed

        self.val_store  = sample_store(top_ks=val_top_ks, test_round_trip=False)
        self.test_store = sample_store(top_ks=test_top_ks, test_round_trip=True)

        self.val_results  = [[] for _ in range(max(self.val_top_ks))]
        self.test_results = [[] for _ in range(max(self.test_top_ks))]

        self.optimizer = optimizer
        self.scheduler = scheduler

        if ckpt_path:
            log.info(f'Loading retro model from {ckpt_path}')
            self.net.load_state_dict(self.load_net(ckpt_path))


    def load_net(self, ckpt_path: str) -> dict[str, Tensor]:
        checkpoint = torch.load(
            ckpt_path,
            map_location=self.device, weights_only=False
        )
        return {
            k.replace('net.', ''): v 
            for k, v in checkpoint['state_dict'].items() 
            if k.startswith('net.')
        }

    def setup(self, stage=None) -> None:
        self.stage = stage
        self.build_torchmetric()
        if self.stage == 'fit':
            log.info(f'\n{self.net}')

    def build_torchmetric(self):
        self.metrics = ('loss', 'loss_X', 'loss_E', 'acc_X', 'acc_E')
        for stage in ('train', 'val', 'test'):
            for metric in self.metrics:
                setattr(self, f"{stage}_{metric}", MeanMetric())
        self.val_best_acc_X = MaxMetric()
        self.val_best_acc_E = MaxMetric()

        
    def step(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx: int):
        loss, log_out = self.step(batch)

        self.log('global_step', self.global_step, on_step=True, on_epoch=False, prog_bar=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=True)

        weight = log_out['bsz']
        for metric in self.metrics:
            result = log_out.get(metric)
            if result:
                getattr(self, f"train_{metric}").update(result, weight)
        
        return {"loss": loss}

    def on_before_optimizer_step(self, optimizer):
        if self.global_rank == 0:
            gn = grad_norm(self.trainer.model, norm_type=2)
            gn = max([v.item() for v in gn.values()]) if len(gn.values()) > 0 else 0
            self.log('train/grad_norm', gn, on_step=True, on_epoch=False, prog_bar=True)
        if (self.global_step + 1) % 100 == 0:
            for metric in self.metrics:
                result = getattr(self, f"train_{metric}")
                if result:
                    self.log(f"train/{metric}", result.compute(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
                    result.reset()
                
    
    def on_after_backward(self) -> None:
        valid_gradients = True
        for _, param in self.trainer.model.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            log.info(f'warning: detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def on_train_epoch_end(self) -> None:
        if dist.is_initialized() and hasattr(self.trainer.datamodule, "train_batch_sampler"):
            self.trainer.datamodule.train_batch_sampler.set_epoch(self.current_epoch + 1)
            self.trainer.datamodule.train_batch_sampler._build_batches()

    # -------# Evaluating #-------- #
    def sampling(self, batch, is_val: bool) -> None:
        raise NotImplementedError()
                
    
    def gather_results(self, is_val: bool) -> tuple[dict[str, float], list]:
        result_store = self.val_store if is_val else self.test_store

        if dist.is_initialized():
            to_log, flattened_samples = self.save_gather(is_val)
        else:
            flattened_samples = result_store.get_samples()
            # Lightning metrics must be numeric during validation.  The
            # upstream ResultStore defaults to human-readable strings such as
            # ``12/64=0.1875``, which crashes every single-GPU validation run.
            # Keep that string form for final test reporting only.
            to_log = result_store.eval(is_str_form=not is_val)
            if is_val:
                log.info(f"Validation sampling metrics: {to_log}")
                for k, v in to_log.items():
                    self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            else:
                log.info(f"Testing sampling metrics: {to_log}")

        # Reset validation results for next epoch
        if is_val:
            result_store.clear()

        return to_log, flattened_samples
    
    def save_gather(self, is_val: bool) -> tuple[dict[str, float], list]:
        assert dist.is_initialized()
        result_store = self.val_store if is_val else self.test_store

        output_dir = HydraConfig.get()['runtime']['output_dir']
        if self.global_rank == 0:
            temp_dir = os.path.join(output_dir, 'gather_temp')
            os.makedirs(temp_dir, exist_ok=True)
            log.info(f"Rank 0: Created temporary directory: {temp_dir}")
        else:
            temp_dir = None
        temp_dir_list = [temp_dir]
        dist.broadcast_object_list(temp_dir_list, src=0)
        temp_dir = temp_dir_list[0]

        log.info(f'Rank: {self.global_rank} Start evaluating!!!!')
        to_log = result_store.eval(
            rank_temp_file=os.path.join(temp_dir, f'rank_{self.global_rank}.jsonl')
        )
        dist.barrier()

        if self.global_rank == 0:
            time.sleep(2)
            to_log = {metric: [0, 0] for metric in to_log.keys()}
            flattened_samples = []
            for rank in range(dist.get_world_size()):
                rank_temp_file = os.path.join(
                    temp_dir, f'rank_{rank}.jsonl'
                )
                rank_metric, rank_samples = result_store.load_results(rank_temp_file)
                for k, (n_acc, n_tol) in rank_metric.items():
                    to_log[k][0] += n_acc
                    to_log[k][1] += n_tol
                flattened_samples.extend(rank_samples)
            
            log.info(f"Rank 0: Start evaluating: {len(flattened_samples)} samples")
            to_log = {
                k: n_acc / n_tol
                for k, (n_acc, n_tol) in to_log.items()
            }
            log.info(f"Rank 0: {to_log}")
            try:
                shutil.rmtree(temp_dir)
                log.info(f"Rank 0: Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                log.warning(f"Rank 0: Failed to clean up temp directory: {e}")

        else:
            to_log = {}
            flattened_samples = []
        
        to_log_list = [to_log] if self.global_rank == 0 else [{}]
        dist.broadcast_object_list(to_log_list, src=0)
        to_log = to_log_list[0]
        dist.barrier()
        
        if is_val:
            log.info(f"Validation sampling metrics: {to_log}")
            for k, v in to_log.items():
                self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            log.info(f"Testing sampling metrics: {to_log}")

        return to_log, flattened_samples


    def on_validation_start(self):
        """Hook called at the beginning of validation."""
        self.val_store.setup_device(self.device)
        val_seed = 2025 + self.global_rank + self.current_epoch * 100
        seed_everything(val_seed, workers=True)
        log.info(f"Validation started with seed: {val_seed}")


    def validation_step(self, batch, batch_idx: int):
        loss, log_out = self.step(batch)
        
        weight = log_out['bsz']
        for metric in self.metrics:
            result = log_out.get(metric)
            if result:
                getattr(self, f"val_{metric}").update(result, weight)

        if batch_idx < 3:
            self.sampling(batch, is_val=True)

        return {"loss": loss}


    def on_validation_epoch_end(self):
        logging_info = {}
        for metric in self.metrics:
            torch_metric: BaseAggregator = getattr(self, f"val_{metric}")
            logging_info[f"val/{metric}"] = torch_metric.compute()
            torch_metric.reset()

        if dist.is_initialized():
            dist.barrier()
        top_ks_log, _ = self.gather_results(is_val=True)
        logging_info.update({f"val/{metric}": v for metric, v in top_ks_log.items()})
        
        for k, v in logging_info.items():
            self.log(f"{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        dict_info = ", ".join(f"{key}={val:.3f}" for key, val in logging_info.items())
        log.info(f"Validation Info @ (Epoch {self.current_epoch}, global step {self.global_step}: {dict_info}")

    
    def on_test_start(self):
        self.test_store.setup_device(self.device)
        seed_everything(self.sampling_seed + self.global_rank, workers=True)


    def test_step(self, batch, batch_idx: int):
        total_batches = self.trainer.num_test_batches[0]
        progress = batch_idx / total_batches
        log.info(f"Test Progress: {batch_idx}/{total_batches} ({progress*100:.2f}%)")

        self.sampling(batch, is_val=False)

    def on_test_end(self):
        top_ks_log, gen_samples = self.gather_results(is_val=False)
        log.info(f"Testing sampling metrics (merged from all ranks): {top_ks_log}")
        
        if self.global_rank == 0:
            hydra_cfg = HydraConfig.get()
            output_dir = hydra_cfg['runtime']['output_dir']
            os.makedirs(output_dir, exist_ok=True)

            gen_sample_json_name = getattr(
                self,
                'gen_sample_json_name',
                'generated_samples.json'
            )
            gen_metric_json_name = getattr(
                self,
                'gen_metric_json_name',
                'generated_metrics.json'
            )

            save_path = os.path.join(output_dir, gen_sample_json_name)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(gen_samples, f, ensure_ascii=False, indent=2)
            log.info(f"Saved merged generated smiles to {save_path}")
            
            metrics_path = os.path.join(output_dir, gen_metric_json_name)
            with open(metrics_path, 'w') as f:
                json.dump(top_ks_log, f, indent=2)
            log.info(f"Saved test metrics to {metrics_path}")

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.net.parameters())
        
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    'interval': 'step',
                    'frequency': 1, # adjust lr_scheduler everytime run evaluation
                },
            }
        return {"optimizer": optimizer}

