"""PyTorch equivalent of FlashAttention's non-interleaved Q/K rotary helper."""

from __future__ import annotations

import torch


def _rotate_non_interleaved(tensor, cos, sin):
    half = cos.shape[-1]
    rotary_dim = 2 * half
    first = tensor[..., :half]
    second = tensor[..., half:rotary_dim]
    remainder = tensor[..., rotary_dim:]
    cos = cos.unsqueeze(0).unsqueeze(2).to(device=tensor.device, dtype=tensor.dtype)
    sin = sin.unsqueeze(0).unsqueeze(2).to(device=tensor.device, dtype=tensor.dtype)
    rotated = torch.cat(
        (first * cos - second * sin, first * sin + second * cos),
        dim=-1,
    )
    if remainder.shape[-1]:
        rotated = torch.cat((rotated, remainder), dim=-1)
    return rotated


def apply_rotary_emb_qkv_(qkv, cos, sin, *_, **__):
    if qkv.ndim != 5 or qkv.shape[2] != 3:
        raise ValueError(f"expected qkv shape [batch, sequence, 3, heads, dim], got {qkv.shape}")
    output = qkv.clone()
    output[:, :, 0] = _rotate_non_interleaved(qkv[:, :, 0], cos, sin)
    output[:, :, 1] = _rotate_non_interleaved(qkv[:, :, 1], cos, sin)
    return output

