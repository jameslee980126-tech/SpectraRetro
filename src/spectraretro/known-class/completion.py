"""Reaction-class conditioning of the SpectraRetro 3D synthon denoiser."""
from contextlib import contextmanager
import torch
from torch import nn
from spectraretro.completion.composition_model import UnknownObservedGraphDDPM, collate_graph, inference_inputs


class ClassResidual(nn.Module):
    """Encode the six reaction classes and add their gated residual to each node."""
    def __init__(self, hidden_nf, class_dim=6, hidden=64, scale=0.1):
        super().__init__()
        self.scale = float(scale)
        self.encode = nn.Sequential(nn.Linear(class_dim, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden), nn.SiLU())
        self.gate = nn.Sequential(nn.Linear(hidden_nf + hidden + 1, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 1), nn.Sigmoid())
        self.projection = nn.Linear(hidden, hidden_nf)

    def forward(self, scalar, reaction_class, time, node_mask):
        encoded = self.encode(reaction_class)
        gate = self.gate(torch.cat((scalar, encoded, time), -1))
        return scalar + self.scale * gate * torch.tanh(self.projection(encoded)) * node_mask


class KnownCompositionDDPM(UnknownObservedGraphDDPM):
    def __init__(self, **kwargs):
        super().__init__(observed_graph_enabled=False, compatibility_enabled=True, **kwargs)
        self._reaction_class = None
        with torch.random.fork_rng(devices=[]):
            self.edm.class_adapter = ClassResidual(kwargs['hidden_nf'])
        self.edm.dynamics.base.dynamics.embedding.register_forward_hook(self._class_hook)

    @staticmethod
    def validate_model_fields(data):
        classes = data.get('reaction_class')
        if classes is not None:
            if classes.ndim != 2 or classes.shape[-1] != 6 or not torch.isfinite(classes).all():
                raise ValueError('Known mode requires finite [B,6] one-hot classes')
            if not torch.all((classes == 0) | (classes == 1)) or not torch.all(classes.sum(-1) == 1):
                raise ValueError('Known classes must be exactly one-hot')
        UnknownObservedGraphDDPM.validate_model_fields({k:v for k,v in data.items() if k!='reaction_class'})

    @contextmanager
    def class_condition(self, data):
        self.validate_model_fields(data)
        classes = data.get('reaction_class')
        if classes is None or classes.shape[0] != data['positions'].shape[0]:
            raise ValueError('One explicit class per known-mode record is required')
        previous = self._reaction_class
        self._reaction_class = classes
        try: yield {k:v for k,v in data.items() if k!='reaction_class'}
        finally: self._reaction_class = previous

    def _class_hook(self, module, inputs, hidden):
        if self._reaction_class is None or self._observed_step is None:
            raise RuntimeError('Class conditioning must be scoped to denoising')
        t, xh, node_mask, context = self._observed_step
        b,n = xh.shape[:2]
        classes = self._reaction_class.to(hidden)[:,None,:].expand(-1,n,-1)
        tt = t.reshape(-1,1,1)
        if tt.shape[0] == 1: tt = tt.expand(b,1,1)
        tt = tt.expand(b,n,1)
        return self.edm.class_adapter(hidden.reshape(b,n,-1),classes,tt,node_mask).reshape_as(hidden)

    def forward(self, data, training):
        with self.class_condition(data) as safe:
            return super().forward(safe, training)

    def sample_chain(self, data, sample_fn=None, keep_frames=None):
        with self.class_condition(data) as safe:
            return super().sample_chain(safe, sample_fn, keep_frames)


def construct(**config):
    return KnownCompositionDDPM(**config)


def prepare_batch(rows, probabilities):
    batch = collate_graph(rows, probabilities)
    classes = torch.tensor([int(row['class_index']) for row in rows])
    batch['reaction_class'] = torch.nn.functional.one_hot(classes, 6).float()
    return batch


def sampling_inputs(batch):
    inputs = inference_inputs(batch)
    inputs['reaction_class'] = batch['reaction_class']
    return inputs
