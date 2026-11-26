"""Class-free observed-synthon graph adapter inside unchanged EDM denoising.

All arms instantiate all adapters. Only the enable flags differ. This module
does not predict bonds, change the EDM loss, or rerank generated candidates.
"""
from contextlib import contextmanager
from pathlib import Path
import os
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
OLD = Path(os.environ.get('RETRO_STAGE2_OLD_ROOT',
    str(HERE.parent / 'old'))).resolve()
CYCLE = Path(os.environ.get('RETRO_STAGE2_CYCLE_ROOT',
    str(HERE.parent / 'cycle'))).resolve()
sys.path.insert(0, str(OLD / 'stage2'))
sys.path.insert(0, str(CYCLE / 'unknown_compatibility'))
from unknown_model import UnknownCompatibilityDDPM
import prior_model
if Path(prior_model.__file__).resolve() != OLD / 'stage2' / 'prior_model.py':
    raise RuntimeError('Prior backbone was imported outside the declared frozen OLD root')

CLASS_FIELDS = {'reaction_class', 'class_index', 'class_id', 'class_metadata_only',
    'reaction_type', 'class_label', 'reaction_class_probabilities'}
BASE_FIELDS = {'uuid', 'num_atoms', 'positions', 'one_hot', 'charges', 'anchors',
    'fragment_mask', 'linker_mask', 'atom_mask', 'edge_mask'}
GRAPH_FIELDS = {'observed_features', 'observed_adjacency', 'observed_graph_available'}
MODEL_FIELDS = BASE_FIELDS | GRAPH_FIELDS | {'composition_probabilities'}


class ObservedGraphAdapter(nn.Module):
    """Two known-bond message passes and a zero-initialized invariant residual."""
    def __init__(self, hidden_nf=128, features=30, width=64, scale=.1):
        super().__init__()
        self.scale = float(scale)
        self.input = nn.Linear(features, width)
        self.messages = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(2)])
        self.updates = nn.ModuleList([nn.Linear(2 * width, width) for _ in range(2)])
        self.projection = nn.Linear(2 * width, hidden_nf)
        self.gate = nn.Sequential(nn.Linear(hidden_nf + 1, width), nn.SiLU(),
            nn.Linear(width, 1), nn.Sigmoid())
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def encode(self, features, adjacency, fragment_mask, anchors):
        known = fragment_mask.to(features)
        adjacency = adjacency.to(features)
        if features.ndim != 3 or features.shape[-1] != 30:
            raise ValueError('Observed features must be [B,N,30]')
        b, n = features.shape[:2]
        if adjacency.shape != (b, n, n) or known.shape != (b, n, 1) or anchors.shape != (b, n, 1):
            raise ValueError('Observed graph and masks have inconsistent shapes')
        if not torch.isfinite(features).all() or not torch.isfinite(adjacency).all():
            raise ValueError('Nonfinite observed graph')
        if not torch.all((known == 0) | (known == 1)):
            raise ValueError('Known mask must be binary')
        if torch.any(features * (1 - known)) or torch.any(adjacency * (1 - known @ known.transpose(1, 2))):
            raise ValueError('Unknown target atoms supplied to observed graph encoder')
        if not torch.all((adjacency == 0) | (adjacency == 1)) or not torch.equal(adjacency, adjacency.transpose(1, 2)):
            raise ValueError('Observed adjacency must be symmetric and binary')
        if torch.any(adjacency.diagonal(dim1=1, dim2=2)):
            raise ValueError('Observed adjacency must have no diagonal edges')
        h = torch.nn.functional.silu(self.input(features)) * known
        denom = adjacency.sum(-1, keepdim=True).clamp_min(1)
        for message, update in zip(self.messages, self.updates):
            neighbor = adjacency @ message(h) / denom
            h = (h + torch.nn.functional.silu(update(torch.cat([h, neighbor], -1)))) * known
        anchor = anchors.to(features) * known
        pooled = (h * anchor).sum(1) / anchor.sum(1).clamp_min(1)
        return h, pooled

    def inject(self, hidden, encoded, pooled, time, mask):
        scalar = torch.cat([encoded, pooled[:, None].expand(-1, encoded.shape[1], -1)], -1)
        residual = self.scale * torch.tanh(self.projection(scalar))
        return hidden + self.gate(torch.cat([hidden, time], -1)) * residual * mask


class UnknownObservedGraphDDPM(UnknownCompatibilityDDPM):
    """Shared-state control/compatibility/graph/combined model; no class conditioning."""
    def __init__(self, observed_graph_enabled=False, compatibility_enabled=False, **kwargs):
        # This isolated factorial experiment permits both separately gated residuals.
        super().__init__(compatibility_enabled=compatibility_enabled, **kwargs)
        self.observed_graph_enabled = bool(observed_graph_enabled)
        self._observed = None
        self._observed_step = None
        with torch.random.fork_rng(devices=[]):
            self.edm.observed_graph_adapter = ObservedGraphAdapter(hidden_nf=kwargs['hidden_nf'])
        dynamics = self.edm.dynamics
        original = dynamics.forward

        def forward(t, xh, node_mask, linker_mask, edge_mask, context):
            previous = self._observed_step
            self._observed_step = (t, xh, node_mask, context)
            try:
                return original(t, xh, node_mask, linker_mask, edge_mask, context)
            finally:
                self._observed_step = previous

        dynamics.forward = forward
        dynamics.base.dynamics.embedding.register_forward_hook(self._observed_embedding_hook)

    @staticmethod
    def validate_model_fields(data):
        if CLASS_FIELDS & set(data):
            raise ValueError('Reaction class is forbidden in unknown-class model input')
        unknown = set(data) - MODEL_FIELDS
        if unknown:
            raise ValueError('Fields outside the model input schema: ' + ','.join(sorted(unknown)))

    def _observed_embedding_hook(self, module, inputs, hidden):
        if not self.observed_graph_enabled:
            return hidden
        if self._observed is None or self._observed_step is None:
            raise RuntimeError('Observed graph requires a scoped denoising condition')
        encoded, pooled, available, known_sizes = self._observed
        t, xh, node_mask, context = self._observed_step
        b, n = xh.shape[:2]
        if b != encoded.shape[0] or torch.any(known_sizes > n):
            raise ValueError('Predicted template does not retain the observed synthon')
        for i, count in enumerate(known_sizes.tolist()):
            if not torch.equal(context[i, :count, -1], torch.ones_like(context[i, :count, -1])):
                raise ValueError('Observed known prefix was not retained by sampling template')
        nodes = hidden.new_zeros(b, n, encoded.shape[-1])
        count = min(n, encoded.shape[1])
        nodes[:, :count] = encoded[:, :count]
        tt = t.reshape(-1, 1, 1)
        if tt.shape[0] == 1:
            tt = tt.expand(b, 1, 1)
        tt = tt.expand(b, n, 1)
        result = self.edm.observed_graph_adapter.inject(hidden.reshape(b, n, -1),
            nodes, pooled, tt, node_mask * available[:, None, None])
        return result.reshape_as(hidden)

    @contextmanager
    def observed_condition(self, data):
        self.validate_model_fields(data)
        old = self._observed
        if self.observed_graph_enabled:
            if not GRAPH_FIELDS <= set(data):
                raise ValueError('Missing observed-synthon graph fields')
            available = data['observed_graph_available']
            if available.shape != (data['positions'].shape[0],):
                raise ValueError('Observed graph availability must be [B]')
            if not torch.all((available == 0) | (available == 1)):
                raise ValueError('Observed availability must be binary')
            features = data['observed_features']
            known = data['fragment_mask']
            expected = (torch.arange(known.shape[1], device=known.device)[None, :, None] <
                known.sum(1).long()[:, None, :]).to(known)
            if not torch.equal(known, expected):
                raise ValueError('Observed atoms must occupy a known prefix')
            if torch.any((features[:, :, :26] - data['one_hot']) * known * available[:, None, None]):
                raise ValueError('Known graph elements disagree with observed element input')
            if torch.any(features * (1 - available[:, None, None].to(features))):
                raise ValueError('Unavailable observed graphs must contain zero features')
            encoded, pooled = self.edm.observed_graph_adapter.encode(features,
                data['observed_adjacency'], known, data['anchors'])
            self._observed = (encoded, pooled, available.to(encoded), known.sum(1).view(-1).long())
        try:
            yield
        finally:
            self._observed = old

    def forward(self, data, training):
        with self.observed_condition(data):
            return super().forward(data, training)

    def sample_chain(self, data, sample_fn=None, keep_frames=None):
        if sample_fn is None:
            raise ValueError('An explicit predicted-size sampler is required')
        self.validate_model_fields(data)
        if __package__:
            from .graph_data import inference_inputs
        else:
            from graph_data import inference_inputs
        safe = inference_inputs(data)
        with self.observed_condition(safe):
            return super().sample_chain(safe, sample_fn, keep_frames)
