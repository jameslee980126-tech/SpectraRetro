"""CPU/PyTorch compatibility subset for RetroDiT's FlashAttention imports.

RetroDiT's GraphDiT path only needs the rotary helper.  The varlen attention
fallback is included so importing the released utility module remains safe on
machines without a matching pre-built ``flash_attn`` wheel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    causal=False,
    **_,
):
    outputs = []
    batch_size = int(cu_seqlens_q.numel() - 1)
    for index in range(batch_size):
        q_start, q_end = (int(value) for value in cu_seqlens_q[index : index + 2])
        k_start, k_end = (int(value) for value in cu_seqlens_k[index : index + 2])
        query = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        key = kv[k_start:k_end, 0].transpose(0, 1).unsqueeze(0)
        value = kv[k_start:k_end, 1].transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=float(dropout_p),
            is_causal=bool(causal),
        )
        outputs.append(attended.squeeze(0).transpose(0, 1))
    if outputs:
        return torch.cat(outputs, dim=0)
    return q.new_empty(q.shape)

