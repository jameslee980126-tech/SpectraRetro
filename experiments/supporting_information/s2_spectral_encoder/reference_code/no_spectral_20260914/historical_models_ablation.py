"""Three product-only encoders with the unchanged Stage1 candidate/scoring task."""
import torch
from torch import nn
from torch.nn import functional as F
from torchdrug import models
from torchdrug.models.fspec_faithful_rc import FSpecFaithfulReactionCenter, _MLP

class PointwiseLayer(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.linear = source.linear
        self.pair_edge_embedding = source.pair_edge_embedding
        self.eps = source.eps
        self.mlp = source.mlp

    def forward(self, eigenvalues, eigenvectors, pair_state, pair_attributes):
        update = self.pair_edge_embedding(self.linear(pair_state), pair_attributes)
        return F.relu(self.mlp(pair_state * (1 + self.eps) + update))

class NoSpectralFSpec(FSpecFaithfulReactionCenter):
    """No spectral mixing/filter. Pair seed, distances, pointwise maps and pooling stay.

    Construct the full module first, then remove frequency encoders, preserving
    exact same-seed initialization of all shared parameters and RNG consumption.
    This still has all-pair distance inputs and node pooling; it is NOT a local GNN.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([PointwiseLayer(layer) for layer in self.layers])

    def _encode_one(self, graph, atom_feature):
        pair_state, attributes = self._pair_seed(graph, atom_feature)
        for layer in self.layers:
            pair_state = layer(None, None, pair_state, attributes)
        pooled = pair_state.mean(dim=1)
        diagonal = pair_state.diagonal(dim1=0, dim2=1).transpose(0, 1)
        node = self.node_pool(pooled + (1 + self.pool_eps) * diagonal)
        endpoints = graph.directed().edge_list[:, :2].long()
        edge = pair_state[endpoints[:, 0], endpoints[:, 1]]
        return node, edge, node.mean(dim=0)

class LocalRGCN(nn.Module):
    """Three-hop RGCN, symmetric local edge readout, no graph-wide pooled inputs."""
    is_pair_center_model = True
    def __init__(self, input_dim, hidden_dims, edge_input_dim, num_relation=4):
        super().__init__()
        self.output_dim = self.edge_feature_dim = hidden_dims[-1]
        self.encoder = models.RGCN(input_dim, hidden_dims, num_relation=num_relation,
            edge_input_dim=edge_input_dim, short_cut=True, batch_norm=False, concat_hidden=False)
        self.edge_readout = _MLP(2*self.output_dim + edge_input_dim, self.output_dim, self.output_dim)

    def forward(self, graph, input, all_loss=None, metric=None):
        out = self.encoder(graph, input, all_loss, metric)
        node = out['node_feature']
        canonical = graph.directed()
        u, v = canonical.edge_list[:, :2].long().unbind(-1)
        edge = self.edge_readout(torch.cat([node[u]+node[v], (node[u]-node[v]).abs(), canonical.edge_feature.float()], -1))
        return {'node_feature':node, 'pair_edge_feature':edge, 'graph_feature':out['graph_feature']}

def make_model(arm, dataset, width=512):
    args = (dataset.node_feature_dim, [width]*3, dataset.edge_feature_dim)
    if arm == 'full_fspec':
        return FSpecFaithfulReactionCenter(*args)
    if arm == 'no_spectral':
        return NoSpectralFSpec(*args)
    if arm == 'local_rgcn':
        return LocalRGCN(*args, num_relation=int(dataset[0]['graph'][1].num_relation))
    raise ValueError(arm)
