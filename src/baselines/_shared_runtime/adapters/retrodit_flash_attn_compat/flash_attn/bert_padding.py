"""Minimal padding helpers matching the FlashAttention API used by RetroDiT."""

from __future__ import annotations

import torch


def unpad_input(hidden_states, attention_mask):
    batch_size, sequence_length = hidden_states.shape[:2]
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, sequence_length),
            dtype=torch.bool,
            device=hidden_states.device,
        )
    attention_mask = attention_mask.bool()
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).flatten()
    flat = hidden_states.reshape(batch_size * sequence_length, *hidden_states.shape[2:])
    unpadded = flat.index_select(0, indices)
    lengths = attention_mask.sum(dim=1, dtype=torch.int32)
    cumulative = torch.zeros(batch_size + 1, dtype=torch.int32, device=hidden_states.device)
    cumulative[1:] = torch.cumsum(lengths, dim=0)
    max_length = int(lengths.max().item()) if batch_size else 0
    return unpadded, indices, cumulative, max_length, lengths


def pad_input(hidden_states, indices, batch, seqlen):
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output.index_copy_(0, indices, hidden_states)
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])

