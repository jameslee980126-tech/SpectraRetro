import os

import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional
from omegaconf import DictConfig

from .extra_features import ExtraFeatures
from src.data.dataset import load_graph_vocab
from src.data.batch_class import DenseGraph

__all__ = [
    'GraphModel'
]

class GraphModel(nn.Module):
    def __init__(
            self,
            graph_vocab: str, 
            fix_product_nodes: bool = False,
            extra_cfg: Optional[DictConfig] = None,
            use_cond: bool = True,
            cond_cfg: Optional[DictConfig] = None,
            **kwargs
        ):
        super().__init__()

        _, x_dec, _, e_dec = load_graph_vocab(graph_vocab)    
        node_dim, edge_dim = len(x_dec), len(e_dec)

        if extra_cfg:
            assert extra_cfg.on_data or extra_cfg.on_noise, \
                "extra_cfg must have on_data or on_noise set to True."
            use_cycles = extra_cfg.get("use_cycles", False)
            use_eigens = extra_cfg.get("use_eigens", False)
            self.sym_E = extra_cfg.get("sym_E", False)
        else:
            use_cycles = False
            use_eigens = False
            self.sym_E = False

        self.extra_on_noise = False
        self.extra_on_data = False
        if use_cycles or use_eigens:
            self.extra_on_noise = extra_cfg.on_noise
            self.extra_on_data  = extra_cfg.on_data

            self.featurizer = ExtraFeatures(
                use_cycles=use_cycles, use_eigens=use_eigens,
                max_n_len=extra_cfg.max_n_len
            )
        else:
            self.featurizer = None

        self.fix_product_nodes = fix_product_nodes

        # compute input/output dimensions
        y_dim = 0
        if cond_cfg and use_cond:
            for v in cond_cfg.values():
                if not v.as_input:
                    continue
                if isinstance(v.k_class, int):
                    y_dim += v.k_class
                else:
                    raise NotImplementedError()


        cond_dim = 2
        self.in_dim_X = node_dim * cond_dim
        self.in_dim_E = edge_dim * cond_dim
        self.in_dim_y = y_dim

        self.out_dim_X = node_dim
        self.out_dim_E = edge_dim

        scale = sum([self.extra_on_noise, self.extra_on_data])
        extra_x_dim = 3 * sum([use_cycles, use_eigens]) * scale
        if use_eigens:
            extra_y_dim = 11
        elif use_cycles:
            extra_y_dim = 5
        else:
            extra_y_dim = 0
        extra_y_dim = extra_y_dim * scale
        extra_e_dim = 0

        self.in_dim_X += extra_x_dim
        self.in_dim_E += extra_e_dim
        self.in_dim_y += extra_y_dim

    def extra_data(
            self,
            *,
            t_E: Tensor, p_E: Tensor, node_mask: Tensor
        ) -> DenseGraph:
        if self.sym_E:
            t_E = 0.5 * (t_E + t_E.transpose(1, 2))
            p_E = 0.5 * (p_E + p_E.transpose(1, 2))

        if self.extra_on_noise:
            dense_noise = self.featurizer(E=t_E, node_mask=node_mask)
            if self.extra_on_data:
                dense_data = self.featurizer(E=p_E, node_mask=node_mask)
                return DenseGraph(
                    X=torch.cat((dense_noise.X, dense_data.X), -1),
                    E=torch.cat((dense_noise.E, dense_data.E), -1),
                    y=torch.cat((dense_noise.y, dense_data.y), -1)
                )
            else:
                return dense_noise
        else:
            if self.extra_on_data:
                return self.featurizer(E=p_E, node_mask=node_mask)
            else:
                return None

    def fix_nodes(
            self,
            pred_X: Tensor, node_mask: Tensor,
            p_X: Tensor, p_mask: Tensor
        ) -> Tensor:
        mask = p_mask.unsqueeze(-1)
        pred_X = pred_X * (~mask) + p_X * mask
        return pred_X * node_mask.unsqueeze(-1)


