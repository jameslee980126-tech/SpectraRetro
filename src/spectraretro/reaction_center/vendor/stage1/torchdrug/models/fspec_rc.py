"""Full-spectrum pair-domain encoder for Stage-1 reaction-center prediction.

This module is an independent TorchDrug implementation inspired by the
pair-domain / bivariate spectral construction of FSpecGNN. It deliberately
does not import the reference repository: the reference code targets PyG
benchmarks and has no reaction-center head. The model keeps a dense
``[num_atom, num_atom, hidden_dim]`` state for one molecule, but applies the
spectral operator as a small-rank product of ordinary graph polynomials. No
``num_atom**2 x num_atom**-squared operator is materialized.
"""

from collections.abc import Sequence

import torch
from torch import nn

from torchdrug import core
from torchdrug.core import Registry as R


def _polynomial_left(signal, laplacian, coefficients):
    """Apply ``sum_k coefficients[k] * L^k`` on the first pair coordinate."""
    result = coefficients[0] * signal
    power = signal
    for coefficient in coefficients[1:]:
        power = torch.matmul(laplacian, power.reshape(power.size(0), -1))
        power = power.reshape_as(signal)
        result = result + coefficient * power
    return result


def _polynomial_right(signal, laplacian, coefficients):
    """Apply ``sum_k coefficients[k] * L^k`` on the second pair coordinate."""
    result = coefficients[0] * signal
    power = signal
    for coefficient in coefficients[1:]:
        # signal[i, j, :] is multiplied by L on its second coordinate:
        # power[i, j, :] = sum_k power[i, k, :] * L[k, j].  ``einsum``
        # makes this contraction explicit and avoids accidental flattening
        # across the atom and channel dimensions.
        power = torch.einsum("ikd,kj->ijd", power, laplacian)
        result = result + coefficient * power
    return result


class _LowRankBivariateLayer(nn.Module):
    """Low-rank bivariate polynomial spectral layer."""

    def __init__(self, hidden_dim, polynomial_order, rank, dropout):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.polynomial_order = int(polynomial_order)
        self.rank = int(rank)
        if self.polynomial_order < 0:
            raise ValueError("polynomial_order must be non-negative")
        if self.rank <= 0:
            raise ValueError("rank must be positive")

        self.left_coeff = nn.Parameter(torch.zeros(self.rank, self.polynomial_order + 1))
        self.right_coeff = nn.Parameter(torch.zeros(self.rank, self.polynomial_order + 1))
        self.left_coeff.data[0, 0] = 1.0
        self.right_coeff.data[0, 0] = 1.0
        self.channel = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, pair_state, laplacian):
        updates = []
        for rank_id in range(self.rank):
            left = _polynomial_left(pair_state, laplacian, self.left_coeff[rank_id])
            updates.append(_polynomial_right(left, laplacian, self.right_coeff[rank_id]))
        update = torch.stack(updates, dim=0).sum(dim=0) / float(self.rank)
        update = self.channel(update)
        pair_state = self.norm(pair_state + torch.tanh(self.gate) * self.dropout(update))
        pair_state = pair_state + self.ffn(self.ffn_norm(pair_state))
        return pair_state


@R.register("models.FSpecReactionCenter")
class FSpecReactionCenter(nn.Module, core.Configurable):
    """Product-only pair-domain encoder for reaction-center identification."""

    is_pair_center_model = True

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None,
                 polynomial_order=2, pair_rank=1, dropout=0.0,
                 readout="mean", **unused):
        super().__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if not hidden_dims or len(set(hidden_dims)) != 1:
            raise ValueError("FSpecReactionCenter requires equal, non-empty hidden dimensions")
        if readout not in {"sum", "mean"}:
            raise ValueError("readout must be `sum` or `mean`")
        self.input_dim = int(input_dim)
        self.output_dim = int(hidden_dims[-1])
        self.edge_feature_dim = self.output_dim
        self.edge_input_dim = int(edge_input_dim or 0)
        self.num_layers = len(hidden_dims)
        self.polynomial_order = int(polynomial_order)
        self.pair_rank = int(pair_rank)
        self.readout = readout

        pair_input_dim = 2 * self.input_dim + self.edge_input_dim + 2
        self.pair_seed = nn.Sequential(
            nn.Linear(pair_input_dim, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )
        self.layers = nn.ModuleList([
            _LowRankBivariateLayer(
                self.output_dim,
                polynomial_order=self.polynomial_order,
                rank=self.pair_rank,
                dropout=dropout,
            )
            for _ in range(self.num_layers)
        ])

    @staticmethod
    def _normalized_laplacian(graph, device, dtype):
        num_atom = int(graph.num_node)
        adjacency = torch.zeros((num_atom, num_atom), device=device, dtype=dtype)
        if graph.edge_list.numel():
            endpoints = graph.edge_list[:, :2].long()
            adjacency[endpoints[:, 0], endpoints[:, 1]] = 1.0
            adjacency[endpoints[:, 1], endpoints[:, 0]] = 1.0
        adjacency.fill_diagonal_(0.0)
        degree = adjacency.sum(dim=-1)
        inv_sqrt_degree = degree.clamp_min(1.0).rsqrt()
        normalized_adjacency = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
        return torch.eye(num_atom, device=device, dtype=dtype) - normalized_adjacency

    def _pair_seed(self, graph, atom_feature):
        num_atom = atom_feature.size(0)
        device, dtype = atom_feature.device, atom_feature.dtype
        edge_dim = self.edge_input_dim
        dense_edge = atom_feature.new_zeros((num_atom, num_atom, edge_dim))
        dense_count = atom_feature.new_zeros((num_atom, num_atom, 1))
        bond_mask = torch.zeros((num_atom, num_atom), device=device, dtype=dtype)
        if graph.edge_list.numel():
            endpoints = graph.edge_list[:, :2].long()
            bond_mask[endpoints[:, 0], endpoints[:, 1]] = 1.0
            bond_mask[endpoints[:, 1], endpoints[:, 0]] = 1.0
            if edge_dim:
                raw_edge = getattr(graph, "edge_feature", None)
                if raw_edge is None or not raw_edge.numel():
                    raw_edge = atom_feature.new_zeros((len(endpoints), edge_dim))
                raw_edge = raw_edge.to(dtype=dtype)
                if raw_edge.size(-1) != edge_dim:
                    adapted = atom_feature.new_zeros((len(endpoints), edge_dim))
                    adapted[:, :min(edge_dim, raw_edge.size(-1))] = raw_edge[:, :edge_dim]
                    raw_edge = adapted
                dense_edge.index_put_((endpoints[:, 0], endpoints[:, 1]), raw_edge, accumulate=True)
                dense_edge.index_put_((endpoints[:, 1], endpoints[:, 0]), raw_edge, accumulate=True)
                ones = atom_feature.new_ones((len(endpoints), 1))
                dense_count.index_put_((endpoints[:, 0], endpoints[:, 1]), ones, accumulate=True)
                dense_count.index_put_((endpoints[:, 1], endpoints[:, 0]), ones, accumulate=True)
                dense_edge = dense_edge / dense_count.clamp_min(1.0)
        diagonal = torch.eye(num_atom, device=device, dtype=dtype)
        left = atom_feature[:, None, :].expand(num_atom, num_atom, -1)
        right = atom_feature[None, :, :].expand(num_atom, num_atom, -1)
        seed_input = torch.cat((left, right, dense_edge, bond_mask.unsqueeze(-1), diagonal.unsqueeze(-1)), dim=-1)
        return self.pair_seed(seed_input)

    def _encode_one(self, graph, atom_feature):
        laplacian = self._normalized_laplacian(graph, atom_feature.device, atom_feature.dtype)
        pair_state = self._pair_seed(graph, atom_feature)
        for layer in self.layers:
            pair_state = layer(pair_state, laplacian)
        pair_state = 0.5 * (pair_state + pair_state.transpose(0, 1))
        node_feature = torch.stack([pair_state[index, index] for index in range(pair_state.size(0))], dim=0)
        canonical = graph.directed()
        if canonical.edge_list.numel():
            edge_index = canonical.edge_list[:, :2].long()
            edge_feature = pair_state[edge_index[:, 0], edge_index[:, 1]]
        else:
            edge_feature = pair_state.new_zeros((0, self.edge_feature_dim))
        graph_feature = node_feature.mean(dim=0) if self.readout == "mean" else node_feature.sum(dim=0)
        return node_feature, edge_feature, graph_feature

    def forward(self, graph, input, all_loss=None, metric=None):
        del all_loss, metric
        if hasattr(graph, "num_nodes"):
            graph_num_nodes = [int(value) for value in graph.num_nodes.tolist()]
            graphs = graph.unpack()
        else:
            graph_num_nodes = [int(graph.num_node)]
            graphs = [graph]
        if sum(graph_num_nodes) != int(input.size(0)):
            raise RuntimeError(
                "FSpecReactionCenter input does not match PackedGraph node counts: "
                f"{input.size(0)} vs {sum(graph_num_nodes)}"
            )

        node_outputs, edge_outputs, graph_outputs = [], [], []
        offset = 0
        for molecule, num_node in zip(graphs, graph_num_nodes):
            node_feature, edge_feature, graph_feature = self._encode_one(
                molecule, input[offset:offset + num_node].float()
            )
            node_outputs.append(node_feature)
            edge_outputs.append(edge_feature)
            graph_outputs.append(graph_feature)
            offset += num_node
        return {
            "node_feature": torch.cat(node_outputs, dim=0),
            "pair_edge_feature": torch.cat(edge_outputs, dim=0),
            "graph_feature": torch.stack(graph_outputs, dim=0),
        }
