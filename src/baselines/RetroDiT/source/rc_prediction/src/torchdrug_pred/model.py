from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torchdrug import models


class DropoutRGCN(models.RGCN):
    """TorchDrug RGCN with optional hidden-state dropout.

    TorchDrug 0.2.1 exposes residual connections and batch normalization on
    ``RGCN`` but has no model-level dropout option.  This subclass preserves
    the native output contract while applying dropout after every relational
    graph convolution (and after a residual addition, when enabled).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_relation: int,
        *,
        short_cut: bool = False,
        batch_norm: bool = False,
        dropout: float = 0.0,
        concat_hidden: bool = True,
        readout: str = "sum",
    ) -> None:
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        super().__init__(
            input_dim=input_dim,
            hidden_dims=list(hidden_dims),
            num_relation=num_relation,
            short_cut=short_cut,
            batch_norm=batch_norm,
            concat_hidden=concat_hidden,
            readout=readout,
        )
        self.dropout_rate = float(dropout)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(
        self,
        graph: Any,
        input: torch.Tensor,
        all_loss: torch.Tensor | None = None,
        metric: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        hiddens = []
        layer_input = input
        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hidden = self.dropout(hidden)
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)
        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature,
        }
