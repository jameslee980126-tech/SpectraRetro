import torch
import torch.nn as nn

from einops import rearrange

from torch import Tensor
from typing import Optional

from ..utils import *

__all__ = [
    'DDiTBlock'
]

class DDiTBlock(nn.Module):
    def __init__(
            self,
            node_dim: int, edge_dim: int, n_heads: int,
            mlp_ratio: int = 4, dropout: float = 0.0,
            rotray_emb: bool = True
        ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = node_dim // self.n_heads

        # Node attention components
        self.attn_qkv = nn.Linear(node_dim, 3 * node_dim, bias=True)
        self.att_out_dense = nn.Linear(node_dim, node_dim, bias=True)
        
        self.x_emb_to_e = nn.Linear(node_dim, edge_dim)
        self.e_emb_to_x_add = nn.Linear(edge_dim, node_dim)
        self.e_emb_to_x_mul = nn.Linear(edge_dim, node_dim)

        
        # Normalization and MLP
        self.x_att_input_norm = nn.LayerNorm(node_dim)
        self.x_mlp_norm = nn.LayerNorm(node_dim)
        self.x_intermediate_mlp = nn.Sequential(
            nn.Linear(node_dim, mlp_ratio * node_dim, bias=True),
            nn.GELU(),
            nn.Linear(mlp_ratio * node_dim, node_dim, bias=True),
        )

        self.e_att_input_norm = nn.LayerNorm(edge_dim)
        self.e_mlp_norm = nn.LayerNorm(edge_dim)
        self.e_intermediate_mlp = nn.Sequential(
            nn.Linear(edge_dim, mlp_ratio * edge_dim, bias=True),
            nn.GELU(),
            nn.Linear(mlp_ratio * edge_dim, edge_dim, bias=True),
        )

        self.dropout = dropout
        self.rotray_emb = rotray_emb
        
    def forward(
            self,
            hidden_states: Tensor, 
            adj_matrix: Tensor,
            attn_mask: Optional[Tensor] = None,
            rotary_cos_sin: Optional[tuple[Tensor, Tensor]] = None,
            sym: bool = False
        ):

        x_mask = attn_mask.unsqueeze(-1)
        e_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(1)

        x_skip = hidden_states
        qkv = self.attn_qkv(self.x_att_input_norm(hidden_states))
        qkv: Tensor = rearrange(
            qkv, "b s (three h d) -> b s three h d", 
            three=3, h=self.n_heads
        )

        if self.rotray_emb:
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        q, k, v = qkv.unbind(dim=2) 
        head_dim = q.shape[-1]
        q = q  / (head_dim ** 0.5)
        e_att: Tensor = torch.einsum('bihd,bjhd->bijhd', q, k)
        e_att = e_att.flatten(3)
        
        e2x_add = self.e_emb_to_x_add(adj_matrix) * e_mask
        e2x_add = e2x_add.view(e_att.shape)
        e2x_mul = self.e_emb_to_x_mul(adj_matrix) * e_mask
        e2x_mul = e2x_mul.view(e_att.shape)

        e_att = e_att * (1 + e2x_mul) + e2x_add
        adj_matrix = adj_matrix + self.x_emb_to_e(e_att)
        if sym:
            adj_matrix = 0.5 * (adj_matrix + adj_matrix.transpose(1, 2))

        e_att = e_att.view(*e_att.shape[:3], self.n_heads, -1)
        scores = e_att.sum(-1).permute(0, 3, 1, 2)

        if attn_mask is not None:
            _attn_mask = attn_mask[:, None, None, :].float()
            _attn_mask  = (1.0 - _attn_mask) * torch.finfo(scores.dtype).min
            scores = scores + _attn_mask
    
        attn_weights = torch.softmax(scores, dim=-1)

        x = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        x = rearrange(x, 'b s h d -> b s (h d)')

        x = self.att_out_dense(x) + x_skip

        x = self.x_intermediate_mlp(self.x_mlp_norm(x)) + x
        adj_matrix = self.e_intermediate_mlp(
            self.e_mlp_norm(adj_matrix)
        ) + adj_matrix
        adj_matrix = adj_matrix * e_mask

        return x, adj_matrix
    
    
