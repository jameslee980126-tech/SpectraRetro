import random

import torch

from torch import Tensor
from torch_geometric.data import Batch, Data

from copy import deepcopy
from omegaconf import DictConfig
from typing import Optional

from .batch_class import *
from .transforms import to_dense
from .dataset import EXCLUDE_KEYS

__all__ = [
    'BaseCollater',
    'GraphCollater',
    'DistributionCollater'
]

class BaseCollater(object):
    def __init__(
            self,
            cond_cfg: Optional[DictConfig] = None,
            **kwargs
        ):

        self.input_cond_dims, self.target_cond_dims = {}, {}
        if cond_cfg:
            for cond_name, v in cond_cfg.items():
                if v.as_input:
                    assert not v.as_target
                    self.add_cond_cfg_(self.input_cond_dims, cond_name, v)
                if v.as_target:
                    assert not v.as_input
                    self.add_cond_cfg_(self.target_cond_dims, cond_name, v)

    def add_cond_cfg_(
            self, cond_dict: dict, cond_name: str, cfg: DictConfig
        ):
        label_v = cfg.k_class
        if isinstance(label_v, int):
            cond_dict[cond_name] = label_v
        else:
            raise NotImplementedError()

    def compute_cond(
            self,
            cond_dims_dict: dict[str, int], batch: Data,
            bsz: int
        ) -> Tensor:
        conds = []
        for cond_name, cond_dim in cond_dims_dict.items():
            cond: Tensor = getattr(batch, cond_name)
            assert cond.size(-1) == cond_dim
            conds.append(cond)
        if conds:
            conds = torch.cat(conds, -1).float()
        else:
            conds = torch.zeros(
                size=(bsz, 0), dtype=torch.float
            )
        return conds

    def __call__(self, batch: list[Data]):
        raise NotImplementedError()


class GraphCollater(BaseCollater):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, batch: list[Data] | list[list[Data]]):
        is_aug_batch = isinstance(batch[0], list)
        if is_aug_batch:
            batch_size = len(batch)
            aug_size = len(batch[0])
            flatten_batch = []
            for b in batch:
                flatten_batch.extend(b)
            batch = flatten_batch
        
        batch = Batch.from_data_list(
            batch, exclude_keys=EXCLUDE_KEYS
        )

        p_X, p_E, p_node_mask = to_dense(
            batch.p_x,
            batch.p_edge_index, batch.p_edge_attr,
            batch.batch
        )

        r_X, r_E, r_node_mask = to_dense(
            batch.r_x,
            batch.r_edge_index, batch.r_edge_attr,
            batch.batch
        )

        assert torch.allclose(r_node_mask, p_node_mask)

        conds = self.compute_cond(
            self.input_cond_dims, batch, bsz=p_X.size(0)
        )
        targets = self.compute_cond(
            self.target_cond_dims, batch, bsz=p_X.size(0)
        )

        p_mask = ~p_X[..., -1].bool()
        assert torch.all(~p_mask <= p_node_mask)

        graph_batch = GraphBatch(
            r_X=r_X, p_X=p_X, r_E=r_E, p_E=p_E,
            node_mask=p_node_mask, p_mask=p_mask,
            cond=conds,
            target=targets
        )

        if is_aug_batch:
            smi_str = batch.p_smiles[::aug_size]
            graph_batch.smi_str = None
            graph_batch = graph_batch.apply(
                lambda x: x.reshape(
                    batch_size, aug_size, *x.shape[1:]
                ).transpose(1, 0)
            )
            graph_batch.aug_batch = True
            aug_graph_batch = []
            for i in range(aug_size):
                aug_graph = graph_batch.apply(lambda x: x[i])
                aug_graph.smi_str = smi_str
                aug_graph_batch.append(aug_graph)
            graph_batch = deepcopy(random.choice(aug_graph_batch))
            return graph_batch, aug_graph_batch
        else:
            return graph_batch


class DistributionCollater(object):
    def __init__(
            self,
            collater: GraphCollater
        ):
        self.collater = collater

    def pop2tensor(
            self, batch: list[list[Data]], key: str
        ) -> Tensor:
        bsz, aug_size = len(batch), len(batch[0])
        data = []
        for aug_b in batch:
            data.extend([b.pop(key) for b in aug_b])
        return torch.tensor(data).reshape(bsz, aug_size)

    def __call__(self, batch: list[list[Data]]):
        perm_weights = self.pop2tensor(batch, 'perm_weight')
        n_perms = self.pop2tensor(batch, 'n_perm')
        assert torch.isclose(n_perms[:, 0, None], n_perms).all()
        base_batch, aug_batches = self.collater(batch)
        n_perms = n_perms[:, 0].clip(max=len(aug_batches))
        return base_batch, aug_batches, perm_weights, n_perms


