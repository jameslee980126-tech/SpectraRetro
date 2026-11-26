"""FSpec v2: ordered pair Route I with an undirected reaction-center readout.

The bivariate layer matches upstream BSGL's ordinary-task parameterization.
Molecular initialization, distance buckets and task heads are explicit adapters.
Old FSpecFaithfulReactionCenter weights belong to v1 and cannot load into v2.
"""
from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from torchdrug import core
from torchdrug.core import Registry as R
from .fspec_faithful_rc import FSpecFaithfulReactionCenter, _PairEdgeEmbedding


class UpstreamMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class BivariateSpectralLayerV2(nn.Module):
    def __init__(self, hidden_dim, edge_input_dim):
        super().__init__()
        self.frequency_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.pair_edge_embedding = _PairEdgeEmbedding(hidden_dim, edge_input_dim)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = UpstreamMLP(hidden_dim, hidden_dim)

    def forward(self, eigenvalues, eigenvectors, pair_state, pair_attributes):
        n = eigenvalues.numel()
        frequencies = torch.stack((eigenvalues[:, None].expand(n, n),
                                   eigenvalues[None, :].expand(n, n)), dim=-1)
        response = self.frequency_encoder(frequencies)
        message = self.pair_edge_embedding(self.linear(pair_state), pair_attributes)
        spectrum = torch.einsum('ij,jkd,kl->ild', eigenvectors.T, message, eigenvectors)
        update = torch.einsum('ij,jkd,kl->ild', eigenvectors, spectrum * response, eigenvectors.T)
        return F.relu(self.mlp((1. + self.eps) * pair_state + update))


@R.register('models.FSpecReactionCenterV2')
class FSpecReactionCenterV2(FSpecFaithfulReactionCenter):
    """Ordered internal states; only the candidate-bond readout is symmetrized."""
    architecture_version = 'fspec_route1_v2'

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None, max_distance=5, dropout=0.0):
        # Do not construct and discard v1 modules / consume their RNG draws.
        nn.Module.__init__(self)
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if not hidden_dims or len(set(hidden_dims)) != 1 or min(hidden_dims) <= 0:
            raise ValueError('hidden_dims must be positive, equal and nonempty')
        if input_dim <= 0 or (edge_input_dim is not None and edge_input_dim < 0) or max_distance < 1:
            raise ValueError('Invalid feature dimensions or max_distance')
        if dropout != 0.:
            raise ValueError('FSpec v2 uses dropout=0: independent spectral dropout violates repeated-eigenspace invariance')
        self.input_dim = int(input_dim)
        self.output_dim = self.edge_feature_dim = int(hidden_dims[0])
        self.edge_input_dim = int(edge_input_dim or 0)
        self.max_distance, self.num_layers = int(max_distance), len(hidden_dims)
        self.pair_seed = nn.Sequential(
            nn.Linear(2 * self.input_dim + self.edge_input_dim + 2, self.output_dim),
            nn.ReLU(), nn.Linear(self.output_dim, self.output_dim))
        # 0..D exact, D+1 reachable beyond D, D+2 disconnected.
        self.distance_embedding = nn.Embedding(self.max_distance + 3, self.output_dim)
        self.layers = nn.ModuleList([BivariateSpectralLayerV2(self.output_dim, self.edge_input_dim)
                                     for _ in hidden_dims])
        # Upstream Eigen_GNN constructs Pool(gin=False, bn=False).
        self.node_pool = UpstreamMLP(self.output_dim, self.output_dim)

    def _shortest_path(self, graph, device):
        n = int(graph.num_node)
        d = torch.full((n, n), float('inf'), device=device)
        d.fill_diagonal_(0)
        ij = graph.edge_list[:, :2].long()
        if ij.numel():
            d[ij[:, 0], ij[:, 1]] = 1
            d[ij[:, 1], ij[:, 0]] = 1
        for k in range(n):
            d = torch.minimum(d, d[:, k, None] + d[None, k, :])
        return torch.where(torch.isfinite(d), d.clamp(max=self.max_distance + 1),
                           torch.full_like(d, self.max_distance + 2)).long()

    def _pair_seed(self, graph, atom_feature):
        n = len(atom_feature)
        attrs, mask = self._pair_attributes(graph, atom_feature)
        pair_input = torch.cat((atom_feature[:, None].expand(n, n, -1),
                                atom_feature[None, :].expand(n, n, -1), attrs,
                                mask[..., None], torch.eye(n, device=atom_feature.device,
                                dtype=atom_feature.dtype)[..., None]), dim=-1)
        return (self.pair_seed(pair_input) +
                self.distance_embedding(self._shortest_path(graph, atom_feature.device))), attrs

    def _encode_one(self, graph, atom_feature):
        state, attrs = self._pair_seed(graph, atom_feature)
        # Double-precision decomposition reduces numerical drift around repeated roots.
        # There is no eigenvector autograd: topology is a fixed product input.
        with torch.no_grad():
            lap = self._normalized_laplacian(graph, atom_feature.device, torch.float64)
            eigenvalues, eigenvectors = torch.linalg.eigh(lap)
            eigenvalues = eigenvalues.to(atom_feature.dtype)
            eigenvectors = eigenvectors.to(atom_feature.dtype)
        for layer in self.layers:
            state = layer(eigenvalues, eigenvectors, state, attrs)
        nodes = F.relu(self.node_pool(state.mean(dim=1)))
        ij = graph.directed().edge_list[:, :2].long()
        edges = .5 * (state[ij[:, 0], ij[:, 1]] + state[ij[:, 1], ij[:, 0]])
        return nodes, edges, nodes.mean(dim=0)
