import math

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn
    import flash_attn.bert_padding as bert_padding
    from flash_attn.layers.rotary import apply_rotary_emb_qkv_
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    # FlashAttention is an optional acceleration dependency.  The benchmark
    # server uses a newer PyTorch/Python stack for which the upstream wheel is
    # unavailable, so retain the same attention and rotary equations through
    # PyTorch's native kernels.
    flash_attn = None
    bert_padding = None
    apply_rotary_emb_qkv_ = None
    _FLASH_ATTN_AVAILABLE = False

from torch import Tensor
from typing import Optional

__all__ = [
    'TimestepEmbedder',
    'PositionalEmbedding',
    'Rotary',
    'PerceiverCrossAttention',
    'apply_rotary_pos_emb'
]

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, n_dim):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_embedding = nn.Embedding(max_seq_len, n_dim)
    
    def forward(self, x: Tensor):
        batch_size, n_seq = x.shape[:2]
        assert n_seq <= self.max_seq_len
        positions = torch.arange(n_seq, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb: Tensor = self.pos_embedding(positions)  # (batch_size, n_seq, n_dim)
        assert x.shape == pos_emb.shape
        return x + pos_emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D)  Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[..., :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)

        return self.cos_cached, self.sin_cached


class PerceiverCrossAttention(nn.Module):
    def __init__(self, dim: int = 3072, dim_head: int = 128, heads: int = 16, kv_dim: int = 2048):
        super().__init__()

        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        # Layer normalization to stabilize training
        self.norm1 = nn.LayerNorm(kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        # Linear transformations to produce queries, keys, and values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.to_out.weight.data.normal_(mean=0.0, std=1e-6)

    def forward(
            self,
            hidden_states: Tensor, cond_embeds: Tensor,
            attn_mask: Optional[Tensor]=None, cond_mask: Optional[Tensor]=None,
            rotary_cos_sin: Optional[Tensor]=None
        ) -> Tensor:
        # Apply layer normalization to the input image and latent features

        cond_embeds = self.norm1(cond_embeds)
        hidden_states = self.norm2(hidden_states)

        # Compute queries, keys, and values
        query = self.to_q(hidden_states)
        key, value = self.to_kv(cond_embeds).chunk(2, dim=-1)

        # Reshape tensors to split into attention heads
        q: Tensor = rearrange(query, "b s (h d) -> b s h d", h=self.heads)
        k: Tensor = rearrange(key,   "b s (h d) -> b s h d", h=self.heads)
        v: Tensor = rearrange(value, "b s (h d) -> b s h d", h=self.heads)

        if rotary_cos_sin is not None:
            cos, sin = rotary_cos_sin
            
            cos_2d = cos[0, :, 0, 0, : cos.size(-1) // 2]
            sin_2d = sin[0, :, 0, 0, : sin.size(-1) // 2]
            
            q_cos, q_sin = cos_2d[:q.size(1)], sin_2d[:q.size(1)]
            k_cos, k_sin = cos_2d[:k.size(1)], sin_2d[:k.size(1)]

            q = _rotary_emb(q, q_cos.to(q), q_sin.to(q))
            k = _rotary_emb(k, k_cos.to(k), k_sin.to(k))

        if _FLASH_ATTN_AVAILABLE:
            flat_q, q_indices, cu_seqlens_q, max_seqlen_q, _ = \
                bert_padding.unpad_input(q, attn_mask.bool() if attn_mask is not None else None)
            flat_kv, _, cu_seqlens_k, max_seqlen_k, _ = \
                bert_padding.unpad_input(torch.stack([k, v], dim=2), cond_mask.bool())
            x = flash_attn.flash_attn_varlen_kvpacked_func(
                q=flat_q, kv=flat_kv,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                dropout_p=0.0, causal=False
            )
            x = bert_padding.pad_input(
                x, q_indices, batch=hidden_states.size(0), seqlen=query.size(1)
            )
        else:
            q_native = q.transpose(1, 2)
            k_native = k.transpose(1, 2)
            v_native = v.transpose(1, 2)
            key_mask = cond_mask.bool()[:, None, None, :] if cond_mask is not None else None
            x = F.scaled_dot_product_attention(
                q_native, k_native, v_native,
                attn_mask=key_mask, dropout_p=0.0, is_causal=False,
            ).transpose(1, 2)
            if attn_mask is not None:
                x = x * attn_mask.bool().unsqueeze(-1).unsqueeze(-1)
        x = rearrange(x, "b s h d -> b s (h d)")

        return self.to_out(x)


def _rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x shape: [batch_size, seq_len, num_heads, head_dim]
    # cos, sin shape: [seq_len, head_dim//2] or [1, seq_len, 1, head_dim//2]
    if cos.dim() == 2:  # [seq_len, head_dim//2]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    
    rotate_dim = cos.shape[-1]
    
    x_rot = x[..., :rotate_dim]
    x_pass = x[..., rotate_dim:]
    
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated_x = torch.cat([-x2, x1], dim=-1)
    x_rot_applied = x_rot * cos + rotated_x * sin
    
    if x_pass.shape[-1] > 0:
        return torch.cat([x_rot_applied, x_pass], dim=-1)
    else:
        return x_rot_applied

def apply_rotary_pos_emb(qkv: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    cos = cos[0, :, 0, 0, :cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, :sin.shape[-1] // 2]
    if _FLASH_ATTN_AVAILABLE:
        return apply_rotary_emb_qkv_(qkv, cos, sin)

    # flash-attn rotates q and k but leaves v unchanged.  cos/sin store half
    # of the rotary dimension; concatenate them to match the split-half
    # convention used by flash-attn's rotary kernel.
    rotary_dim = 2 * cos.shape[-1]
    cos_full = torch.cat((cos, cos), dim=-1)[None, :, None, :].to(qkv)
    sin_full = torch.cat((sin, sin), dim=-1)[None, :, None, :].to(qkv)

    def rotate(x: Tensor) -> Tensor:
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        first, second = x_rot.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        x_rot = x_rot * cos_full + rotated * sin_full
        return torch.cat((x_rot, x_pass), dim=-1) if x_pass.shape[-1] else x_rot

    q, k, v = qkv.unbind(dim=2)
    return torch.stack((rotate(q), rotate(k), v), dim=2)

