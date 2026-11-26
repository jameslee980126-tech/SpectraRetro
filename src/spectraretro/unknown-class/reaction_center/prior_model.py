"""Unknown-class FSpec center model with the same optional chemical adapter.

Reaction classes remain metadata for reporting only.  They are not included in
the TorchDrug task feature tuple and are never read by ``predict``.
"""
from spectraretro.reaction_center.features import (
    ROOT, VENDOR, SCHEMA, NODE_COLUMNS, BOND_COLUMNS, NODE_DIM, EDGE_DIM,
    CandidatePriorAdapter, attach_product_prior, describe_product,
)
import torch
from torch import nn
from torchdrug import layers, models, tasks


class UnknownClassPriorCenter(tasks.CenterIdentification):
    """FSpec candidate readout with no reaction-class input or side channel."""

    def __init__(self, model, prior=False, prior_hidden=64, residual_limit=1.):
        super().__init__(model, None, feature=("graph", "atom", "bond"),
                         mlp_hidden_dim=model.output_dim, fusion_mode="raw")
        self.use_prior = bool(prior)
        self.prior_hidden = int(prior_hidden)
        self.residual_limit = float(residual_limit)

    def preprocess(self, train_set, valid_set, test_set):
        if test_set:
            raise ValueError("Test may not be loaded during model construction")
        super().preprocess(train_set, valid_set, [])
        self.edge_mlp = layers.MLP(self.model.edge_feature_dim,
                                   [self.mlp_hidden_dim, 1], activation=self.mlp_act)
        self.node_mlp = layers.MLP(self.model.output_dim,
                                   [self.mlp_hidden_dim, 1], activation=self.mlp_act)
        if self.use_prior:
            state = torch.get_rng_state()
            self.edge_prior = CandidatePriorAdapter(
                self.model.edge_feature_dim, EDGE_DIM, 0,
                self.prior_hidden, self.residual_limit)
            self.node_prior = CandidatePriorAdapter(
                self.model.output_dim, NODE_DIM, 0,
                self.prior_hidden, self.residual_limit)
            torch.set_rng_state(state)

    def predict(self, batch, all_loss=None, metric=None):
        product = batch["graph"][1]
        graph = product.directed()
        output = self.model(product, product.node_feature.float(), all_loss, metric)
        nodes, edges = output["node_feature"], output["pair_edge_feature"]
        if self.use_prior:
            node_classes = nodes.new_zeros(nodes.shape[0], 0)
            edge_classes = edges.new_zeros(edges.shape[0], 0)
            nodes, ngate, nres = self.node_prior(
                nodes, graph.chemical_prior_node,
                graph.chemical_prior_node_mask, node_classes)
            edges, egate, eres = self.edge_prior(
                edges, graph.chemical_prior_edge,
                graph.chemical_prior_edge_mask, edge_classes)
            self.last_prior = dict(node_gate=ngate.detach(), edge_gate=egate.detach(),
                                   node_residual=nres.detach(), edge_residual=eres.detach())
        return self._collate(self.edge_mlp(edges).squeeze(-1),
                             self.node_mlp(nodes).squeeze(-1), graph)


def build_task(train, valid, prior=False, width=512, num_layers=3):
    sample = train[0]["graph"][1]
    model = models.FSpecFaithfulReactionCenter(
        sample.node_feature.shape[-1], [width] * num_layers,
        edge_input_dim=sample.edge_feature.shape[-1], dropout=0.)
    task = UnknownClassPriorCenter(model, prior=prior)
    task.preprocess(train, valid, [])
    return task
