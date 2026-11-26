"""Task-faithful adaptation of the official FSpec bivariate spectral layer.

The upstream FSpecGNN code is written for padded PyG batches and node / graph
benchmarks.  This module keeps its central operation -- encode eigenvalue pairs,
transform a pair state into the Laplacian eigenbasis, apply a bivariate filter,
and transform back -- but exposes TorchDrug's variable-size molecular graphs
and the canonical Stage-1 bond candidates.

This is deliberately a clean product-only baseline.  It does not construct a
planar dual, use reaction classes, or add hand-designed chemical features.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core
from torchdrug.core import Registry as R


class _MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, value):
        return self.network(value)


class _PairEdgeEmbedding(nn.Module):
    """Official EdgeEmbedding analogue for dense pair attributes."""

    def __init__(self, hidden_dim, edge_input_dim, dropout=0.0):
        super().__init__()
        self.edge_input_dim = int(edge_input_dim)
        self.edge_encoder = (
            _MLP(self.edge_input_dim, hidden_dim, hidden_dim, dropout=dropout)
            if self.edge_input_dim > 0 else None
        )

    def forward(self, message, attributes):
        if self.edge_encoder is not None:
            message = message + self.edge_encoder(attributes)
        return F.relu(message)


class _OfficialBivariateSpectralLayer(nn.Module):
    """The official FSpec-style eigenbasis filter for one molecule."""

    def __init__(self, hidden_dim, edge_input_dim, dropout=0.0):
        super().__init__()
        self.frequency_encoder = _MLP(2, hidden_dim, hidden_dim, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.pair_edge_embedding = _PairEdgeEmbedding(
            hidden_dim, edge_input_dim, dropout=dropout
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = _MLP(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)

    def forward(self, eigenvalues, eigenvectors, pair_state, pair_attributes):
        # E_i, E_j -> a learned bivariate frequency response f(E_i, E_j).
        num_atom = eigenvalues.numel()
        frequency_pair = torch.stack(
            (
                eigenvalues[:, None].expand(-1, num_atom),
                eigenvalues[None, :].expand(num_atom, -1),
            ),
            dim=-1,
        )
        frequency_filter = self.frequency_encoder(frequency_pair)
        # Stage-1 candidates are undirected chemical bonds.  The upstream
        # layer is defined on ordered pair tokens, so its learned response is
        # symmetrized here to keep the bond representation independent of the
        # arbitrary edge orientation emitted by a molecular graph parser.
        frequency_filter = 0.5 * (
            frequency_filter + frequency_filter.transpose(0, 1)
        )

        # Encode dense edge / bond attributes in the pair domain.  Non-bond
        # entries are zero-filled by the caller and remain valid pair context.
        pair_message = self.linear(pair_state)
        pair_message = self.pair_edge_embedding(pair_message, pair_attributes)

        # U^T S U, then elementwise bivariate spectral filtering, then U S U^T.
        spectral_state = torch.einsum(
            "ij,jkd,kl->ild",
            eigenvectors.transpose(0, 1),
            pair_message,
            eigenvectors,
        )
        spectral_state = spectral_state * frequency_filter
        pair_update = torch.einsum(
            "ij,jkd,kl->ild",
            eigenvectors,
            spectral_state,
            eigenvectors.transpose(0, 1),
        )

        # No zero residual gate is used here.  The official layer's learnable
        # epsilon keeps an identity path while allowing spectral parameters to
        # receive gradients from the first backward pass.
        pair_state = self.mlp(pair_state * (1.0 + self.eps) + pair_update)
        return F.relu(pair_state)


@R.register("models.FSpecFaithfulReactionCenter")
class FSpecFaithfulReactionCenter(nn.Module, core.Configurable):
    """Official-FSpec-style product-only encoder for Stage-1 bond ranking."""

    is_pair_center_model = True

    def __init__(
        self,
        input_dim,
        hidden_dims,
        edge_input_dim=None,
        max_distance=5,
        dropout=0.0,
        **unused,
    ):
        super().__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if not hidden_dims or len(set(hidden_dims)) != 1:
            raise ValueError(
                "FSpecFaithfulReactionCenter requires equal, non-empty hidden dimensions"
            )
        if int(max_distance) < 0:
            raise ValueError("max_distance must be non-negative")

        self.input_dim = int(input_dim)
        self.output_dim = int(hidden_dims[-1])
        self.edge_feature_dim = self.output_dim
        self.edge_input_dim = int(edge_input_dim or 0)
        self.max_distance = int(max_distance)
        self.num_layers = len(hidden_dims)

        # Symmetric endpoint construction makes bond scores invariant to
        # swapping the two atoms of a candidate bond.
        pair_input_dim = 2 * self.input_dim + self.edge_input_dim + 2
        self.pair_seed = _MLP(
            pair_input_dim,
            self.output_dim,
            self.output_dim,
            dropout=dropout,
        )
        self.distance_embedding = nn.Embedding(self.max_distance + 1, self.output_dim)
        self.layers = nn.ModuleList(
            [
                _OfficialBivariateSpectralLayer(
                    self.output_dim,
                    self.edge_input_dim,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.pool_eps = nn.Parameter(torch.zeros(1))
        self.node_pool = _MLP(
            self.output_dim,
            self.output_dim,
            self.output_dim,
            dropout=dropout,
        )

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
        normalized_adjacency = (
            inv_sqrt_degree[:, None]
            * adjacency
            * inv_sqrt_degree[None, :]
        )
        return torch.eye(num_atom, device=device, dtype=dtype) - normalized_adjacency

    def _shortest_path(self, graph, device):
        num_atom = int(graph.num_node)
        distance = torch.full(
            (num_atom, num_atom),
            float("inf"),
            device=device,
            dtype=torch.float32,
        )
        diagonal = torch.arange(num_atom, device=device)
        distance[diagonal, diagonal] = 0.0
        if graph.edge_list.numel():
            endpoints = graph.edge_list[:, :2].long()
            distance[endpoints[:, 0], endpoints[:, 1]] = 1.0
            distance[endpoints[:, 1], endpoints[:, 0]] = 1.0
        # Molecular products are small; Floyd--Warshall is deterministic and
        # keeps the pair relation inside the same tensor contract as FSpec.
        for pivot in range(num_atom):
            distance = torch.minimum(
                distance,
                distance[:, pivot, None] + distance[None, pivot, :],
            )
        distance = torch.nan_to_num(
            distance,
            nan=float(self.max_distance),
            posinf=float(self.max_distance),
        )
        return distance.long()

    def _pair_attributes(self, graph, atom_feature):
        num_atom = int(atom_feature.size(0))
        dtype = atom_feature.dtype
        edge_dim = self.edge_input_dim
        dense_edge = atom_feature.new_zeros((num_atom, num_atom, edge_dim))
        dense_count = atom_feature.new_zeros((num_atom, num_atom, 1))
        bond_mask = atom_feature.new_zeros((num_atom, num_atom))
        raw_edge = getattr(graph, "edge_feature", None)
        if edge_dim:
            expected = (len(graph.edge_list), edge_dim)
            if raw_edge is None or tuple(raw_edge.shape) != expected:
                actual = None if raw_edge is None else tuple(raw_edge.shape)
                raise ValueError(
                    f"Stage1 edge_feature shape mismatch: expected {expected}, got {actual}. "
                    "Rebuild the feature cache with the checkpoint's feature schema; "
                    "implicit padding/truncation is forbidden."
                )
            if raw_edge.is_floating_point() and not torch.isfinite(raw_edge).all():
                raise ValueError("Stage1 edge_feature contains nonfinite values")
            raw_edge = raw_edge.to(device=atom_feature.device, dtype=dtype)
        if graph.edge_list.numel():
            endpoints = graph.edge_list[:, :2].long()
            bond_mask[endpoints[:, 0], endpoints[:, 1]] = 1.0
            bond_mask[endpoints[:, 1], endpoints[:, 0]] = 1.0
            if edge_dim:
                dense_edge.index_put_(
                    (endpoints[:, 0], endpoints[:, 1]), raw_edge, accumulate=True
                )
                dense_edge.index_put_(
                    (endpoints[:, 1], endpoints[:, 0]), raw_edge, accumulate=True
                )
                ones = atom_feature.new_ones((len(endpoints), 1))
                dense_count.index_put_(
                    (endpoints[:, 0], endpoints[:, 1]), ones, accumulate=True
                )
                dense_count.index_put_(
                    (endpoints[:, 1], endpoints[:, 0]), ones, accumulate=True
                )
                dense_edge = dense_edge / dense_count.clamp_min(1.0)
        return dense_edge, bond_mask

    def _pair_seed(self, graph, atom_feature):
        num_atom = int(atom_feature.size(0))
        left = atom_feature[:, None, :].expand(num_atom, num_atom, -1)
        right = atom_feature[None, :, :].expand(num_atom, num_atom, -1)
        dense_edge, bond_mask = self._pair_attributes(graph, atom_feature)
        diagonal = torch.eye(
            num_atom, device=atom_feature.device, dtype=atom_feature.dtype
        )
        pair_input = torch.cat(
            (
                left + right,
                torch.abs(left - right),
                dense_edge,
                bond_mask.unsqueeze(-1),
                diagonal.unsqueeze(-1),
            ),
            dim=-1,
        )
        pair_state = self.pair_seed(pair_input)
        distance = self._distance_indices(graph, atom_feature.device)
        pair_state = pair_state + self.distance_embedding(distance)
        return pair_state, dense_edge

    def _distance_indices(self, graph, device):
        return self._shortest_path(graph, device).clamp(0, self.max_distance)

    def _encode_one(self, graph, atom_feature):
        pair_state, pair_attributes = self._pair_seed(graph, atom_feature)
        laplacian = self._normalized_laplacian(
            graph, atom_feature.device, atom_feature.dtype
        )
        # The molecular graph is fixed during training.  Keeping the spectral
        # decomposition out of autograd avoids undefined eigenvector gradients
        # at repeated Laplacian eigenvalues; learnable gradients still flow
        # through the frequency encoder and pair-state transforms.
        with torch.no_grad():
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        for layer in self.layers:
            pair_state = layer(
                eigenvalues,
                eigenvectors,
                pair_state,
                pair_attributes,
            )
        # Official vertex pooling averages one pair coordinate and adds the
        # diagonal.  The task consumes both node tokens and candidate-bond
        # tokens, so we retain the full pair state for edges.
        node_pool = pair_state.mean(dim=1)
        diagonal = pair_state.diagonal(dim1=0, dim2=1).transpose(0, 1)
        node_feature = self.node_pool(
            node_pool + (1.0 + self.pool_eps) * diagonal
        )
        canonical = graph.directed()
        if canonical.edge_list.numel():
            edge_index = canonical.edge_list[:, :2].long()
            edge_feature = pair_state[edge_index[:, 0], edge_index[:, 1]]
        else:
            edge_feature = pair_state.new_zeros((0, self.edge_feature_dim))
        graph_feature = node_feature.mean(dim=0)
        return node_feature, edge_feature, graph_feature

    def forward(self, graph, input, all_loss=None, metric=None):
        del all_loss, metric
        if input.ndim != 2 or input.shape[1] != self.input_dim:
            raise ValueError(f"Stage1 node_feature must have shape [N, {self.input_dim}], got {tuple(input.shape)}")
        if not torch.isfinite(input).all():
            raise ValueError("Stage1 node_feature contains nonfinite values")
        if hasattr(graph, "num_nodes"):
            graph_num_nodes = [int(value) for value in graph.num_nodes.tolist()]
            graphs = graph.unpack()
        else:
            graph_num_nodes = [int(graph.num_node)]
            graphs = [graph]
        if sum(graph_num_nodes) != int(input.size(0)):
            raise RuntimeError(
                "FSpecFaithfulReactionCenter input does not match PackedGraph node counts: "
                f"{input.size(0)} vs {sum(graph_num_nodes)}"
            )
        if not graph_num_nodes or any(n <= 0 for n in graph_num_nodes):
            raise ValueError("Stage1 requires nonempty molecular graphs")

        node_outputs, edge_outputs, graph_outputs = [], [], []
        offset = 0
        for molecule, num_node in zip(graphs, graph_num_nodes):
            node_feature, edge_feature, graph_feature = self._encode_one(
                molecule, input[offset : offset + num_node].float()
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
