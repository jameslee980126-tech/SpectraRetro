"""Parameter-matched finite-neighborhood alternative to bivariate FSpec.

Only the pair propagation is changed. Input features, pair initialization and
task readout are inherited from the immutable training snapshot.
"""
import torch
from torch.nn import functional as F
from torchdrug.models.fspec_faithful_rc import (
    FSpecFaithfulReactionCenter, _OfficialBivariateSpectralLayer,
)


class LocalPairLayer(_OfficialBivariateSpectralLayer):
    def propagate(self, adjacency, message):
        # Keep the original parameter budget; evaluate the same-shaped MLP on
        # fixed local-basis IDs rather than all molecular eigenvalue pairs.
        basis_ids = message.new_tensor([[0., 0.], [0., 1.], [1., 1.]])
        weights = self.frequency_encoder(basis_ids)
        left = torch.einsum('ij,jkd->ikd', adjacency, message)
        right = torch.einsum('ijd,jk->ikd', message, adjacency)
        both = torch.einsum('ijd,jk->ikd', left, adjacency)
        return (weights[0] * message + weights[1] * (left + right) * .5
                + weights[2] * both)

    def forward(self, adjacency, pair_state, pair_attributes):
        message = self.pair_edge_embedding(self.linear(pair_state), pair_attributes)
        update = self.propagate(adjacency, message)
        return F.relu(self.mlp(pair_state * (1. + self.eps) + update))


class LocalPairReactionCenter(FSpecFaithfulReactionCenter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No new construction / RNG draws: parameters and their initialization
        # stay byte-identical to FSpec under a paired seed.
        for layer in self.layers:
            layer.__class__ = LocalPairLayer

    def _encode_one(self, graph, atom_feature):
        pair_state, pair_attributes = self._pair_seed(graph, atom_feature)
        laplacian = self._normalized_laplacian(graph, atom_feature.device, atom_feature.dtype)
        adjacency = torch.eye(int(graph.num_node), device=atom_feature.device,
                              dtype=atom_feature.dtype) - laplacian
        for layer in self.layers:
            pair_state = layer(adjacency, pair_state, pair_attributes)
        # Identical output pooling and candidate extraction to frozen FSpec.
        node_pool = pair_state.mean(dim=1)
        diagonal = pair_state.diagonal(dim1=0, dim2=1).transpose(0, 1)
        node_feature = self.node_pool(node_pool + (1. + self.pool_eps) * diagonal)
        canonical = graph.directed()
        if canonical.edge_list.numel():
            edge_index = canonical.edge_list[:, :2].long()
            edge_feature = pair_state[edge_index[:, 0], edge_index[:, 1]]
        else:
            edge_feature = pair_state.new_zeros((0, self.edge_feature_dim))
        graph_feature = node_feature.mean(dim=0)
        return node_feature, edge_feature, graph_feature


def make_model(arm, dataset, width=512):
    cls = {'full_fspec': FSpecFaithfulReactionCenter,
           'local_pair': LocalPairReactionCenter}[arm]
    return cls(dataset.node_feature_dim, [width] * 3, dataset.edge_feature_dim)
