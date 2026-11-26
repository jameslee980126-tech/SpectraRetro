from omegaconf import DictConfig
from functools import partial
from typing import Mapping, Optional, Sequence, Union

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import lightning.pytorch as pl

from .dataset import DatasetModule
from .collater import GraphCollater
from .p_sampler import ProbSampler

__all__ = ['DataModule']

class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root: str, batch_size: int, num_workers: int, 

            n_dummy: int, max_n_len: int,
            graph_vocab: str,

            shuffle_order: bool = False, canonicalize: bool = True,
            dn_last: bool = True, 

            sampler_cfg: Optional[DictConfig] = None,
            cond_cfg: Optional[DictConfig] = None,

            perm_prob_sampler: partial[ProbSampler] = None,
            val_top_ks: Sequence[int] = [1, 3],
            test_top_ks: Sequence[int] = [1, 3, 5, 10],
            perm_types: Union[list[str], str] = None,
            external_perm_paths: Optional[Mapping[str, Optional[str]]] = None,
            collater = None,
            **kwargs
        ):
        if canonicalize:
            assert not shuffle_order
            
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dn_last = dn_last

        self.collater = GraphCollater(cond_cfg=cond_cfg)
        if collater is not None:
            self.collater = collater(collater=self.collater)

        self.sampler_cfg = sampler_cfg

        is_leaky = False
        if n_dummy <= 0:
            is_leaky = True
        if not canonicalize and not shuffle_order:
            is_leaky = True

        self.data_kwargs = {
            'root': data_root, 'graph_vocab': graph_vocab,

            # dataset settings
            'is_leaky': is_leaky, 'max_n_len': max_n_len, 'n_dummy': n_dummy,
            'shuffle_order': shuffle_order, 'canonicalize': canonicalize,
            'dn_last': dn_last,

            'perm_prob_sampler': perm_prob_sampler,
            'val_top_k': max(val_top_ks), 'test_top_k': max(test_top_ks),
            'perm_types': perm_types,
            'external_perm_paths': external_perm_paths,
        }

        self.train_dataset: DatasetModule | None
        self.valid_dataset: DatasetModule | None
        self.test_dataset: DatasetModule | None

    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError()

    def dataloader(self, dataset):
        return setup_dataloader(
            dataset, 
            self.collater,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            sampler_cfg=self.sampler_cfg
        )

    def train_dataloader(self):
        return self.dataloader(self.train_dataset)

    def val_dataloader(self):
        return self.dataloader(self.valid_dataset)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=self.collater,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            sampler=ExactDistributedTestSampler(self.test_dataset),
            pin_memory=True,
        )


class ExactDistributedTestSampler(DistributedSampler):
    """Shard test indices across ranks without shuffling or padding."""

    def __init__(self, dataset):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        super().__init__(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)

def setup_dataloader(
        ds: DatasetModule, 
        collater,
        batch_size: int, num_workers: int, shuffle: bool,
        sampler_cfg: Optional[DictConfig] = None
    ) -> DataLoader:
    if sampler_cfg and sampler_cfg.use_sampler:
        raise NotImplementedError()
    else:
        return DataLoader(
            dataset=ds, 
            collate_fn=collater, batch_size=batch_size,
            num_workers=num_workers, shuffle=shuffle, pin_memory=True
        )


