import torch
import torch.nn as nn

from torch import Tensor
from typing import Type
from omegaconf import DictConfig

import importlib

from ..utils import *
from ..graph_model import GraphModel

__all__ = ['GraphDiT']


class GraphDiT(GraphModel):
    def __init__(
            self,
            x_dim: int, e_dim: int,
            n_layers: int, n_heads: int,
            dropout: float,
            block_cfg: DictConfig,
            n_dummy: int = 10,
            max_n_len: int = 100, 
            symmetry_E: bool = True,
            rotray_emb: bool = True, pos_emb: bool = False,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.n_layers = n_layers

        in_dim_X, in_dim_E, in_dim_y = self.in_dim_X, self.in_dim_E, self.in_dim_y
        
        self.symmetry_E = symmetry_E

        # Input projection
        self.node_input = nn.Linear(in_dim_X, x_dim)
        self.edge_input = nn.Linear(in_dim_E, e_dim)
        
        # Condition embedding projection
        self.t_embed = TimestepEmbedder(x_dim)
        self.y_embed = nn.Linear(in_dim_y, x_dim)
        self.rotary_emb = Rotary(x_dim // n_heads)
        if pos_emb:
            self.pos_emb = PositionalEmbedding(max_n_len + n_dummy, in_dim_X)
        else:
            self.pos_emb = None
        
        block_class = get_class_from_target(block_cfg.module)
        extra_kwargs = {k: v for k, v in block_cfg.items() if k != 'module'}
        
        self.layers = nn.ModuleList([
            block_class(
                node_dim=x_dim, edge_dim=e_dim, n_heads=n_heads,
                dropout=dropout, rotray_emb=rotray_emb, **extra_kwargs
            )
            for _ in range(n_layers)
        ])

        # Output normalization
        self.out_norm_node = nn.LayerNorm(x_dim)
        self.out_norm_edge = nn.LayerNorm(e_dim)
        
        # Optional: output projection
        self.output_proj_node = nn.Linear(x_dim, self.out_dim_X)
        if self.out_dim_E > 0:
            self.output_proj_edge = nn.Linear(e_dim, self.out_dim_E)
        else:
            self.output_proj_edge = None

    def model_forward(
            self,
            node_features, adj_matrix,
            t, y, attention_mask,
            **kwargs
        ):
        """
        node_features: [batch_size, n_nodes, node_dim]
        adj_matrix: [batch_size, n_nodes, n_nodes] - dense adjacency matrix
        """

        if self.symmetry_E:
            adj_matrix = 0.5 * (adj_matrix + adj_matrix.transpose(-3, -2))

        cond_embed: Tensor = self.t_embed(t) + self.y_embed(y)
        assert cond_embed.ndim == 2

        cond_embed = cond_embed.unsqueeze(1)
        assert cond_embed.ndim == 3

        # Project input features
        hidden_x = self.node_input(node_features)
        hidden_e = self.edge_input(adj_matrix)
        
        # Project and add condition embedding to each node
        hidden_x = hidden_x + cond_embed

        rotary_cos_sin = self.rotary_emb(hidden_x)
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_x, hidden_e = layer(
                hidden_x, hidden_e, attention_mask,
                rotary_cos_sin, sym=self.symmetry_E
            )
        
        # Output projection
        node_embeddings = self.output_proj_node(hidden_x)
        x_mask = attention_mask.unsqueeze(-1)
        node_embeddings = node_embeddings * x_mask

        if self.output_proj_edge:
            edge_embeddings = self.output_proj_edge(hidden_e)
            if self.symmetry_E:
                edge_embeddings = 0.5 * (edge_embeddings + edge_embeddings.transpose(-3, -2))
            e_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(1)
            edge_embeddings = edge_embeddings * e_mask
            return {'X': node_embeddings, 'E': edge_embeddings}
        else:
            return {'X': node_embeddings}


    def forward(
            self,
            t: Tensor, node_mask: Tensor, cond: Tensor, 
            t_X: Tensor, t_E: Tensor,
            p_X: Tensor, p_E: Tensor,
            p_mask: Tensor,
            **kwargs
        ) -> dict[str, Tensor]:
        assert p_X is not None and p_E is not None
        input_X = torch.cat((p_X, t_X), -1)
        input_E = torch.cat((p_E, t_E), -1)

        extra_data = self.extra_data(
            t_E=t_E, p_E=p_E, node_mask=node_mask
        )
        if extra_data:
            input_X = torch.cat((input_X, extra_data.X), -1)
            input_E = torch.cat((input_E, extra_data.E), -1)
            cond = torch.cat((cond, extra_data.y), -1)

        if self.pos_emb:
            input_X = self.pos_emb(input_X)
            
        pred = self.model_forward(
            input_X, input_E, t, cond, node_mask,
        )

        if self.fix_product_nodes:
            pred['X'] = self.fix_nodes(
                pred['X'], node_mask, p_X, p_mask
            )

        return pred


def get_class_from_target(target_path: str) -> Type:
    module_path, class_name = target_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

