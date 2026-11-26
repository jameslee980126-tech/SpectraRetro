"""Class-free schema, retaining observed graphs and isolating training targets."""
import torch
try:
    from .graph_model import BASE_FIELDS, GRAPH_FIELDS
except ImportError:
    from graph_model import BASE_FIELDS, GRAPH_FIELDS
from src.datasets import collate

ROW_FIELDS = {'uuid', 'num_atoms', 'positions', 'one_hot', 'charges', 'anchors',
    'fragment_mask', 'linker_mask'}


def collate_graph(items, probabilities=None):
    """Raw cache class/geometry labels are metadata only and are never collated."""
    if not items:
        raise ValueError('Cannot collate an empty batch')
    if any(not ROW_FIELDS <= set(x) or not GRAPH_FIELDS <= set(x) for x in items):
        raise ValueError('Missing base or observed graph cache fields')
    batch = collate([{k: v for k, v in x.items() if k in ROW_FIELDS} for x in items])
    b, n = batch['positions'].shape[:2]
    batch['observed_features'] = torch.nn.utils.rnn.pad_sequence(
        [x['observed_features'] for x in items], batch_first=True)
    batch['observed_adjacency'] = torch.zeros(b, n, n, dtype=torch.bool)
    batch['observed_graph_available'] = torch.tensor(
        [bool(x['observed_graph_available']) for x in items], dtype=torch.bool)
    for i, x in enumerate(items):
        ni = len(x['positions'])
        if x['observed_features'].shape != (ni, 30) or x['observed_adjacency'].shape != (ni, ni):
            raise ValueError('Observed cache shape differs from base atom shape')
        batch['observed_adjacency'][i, :ni, :ni] = x['observed_adjacency']
    if probabilities is not None:
        p = torch.as_tensor(probabilities, dtype=torch.float32)
        if p.shape != (b, 26):
            raise ValueError('Compatibility probabilities must be [B,26]')
        batch['composition_probabilities'] = p
    return batch


def inference_inputs(batch):
    """Remove target attributes and target-size footprints before size sampling.

    Known atoms are a prefix. All arrays are cropped to the maximum observed
    length, missing masks are cleared, and masks/counts are recomputed. The
    original vendor sampler builds its own predicted-size templates afterwards.
    """
    allowed = BASE_FIELDS | GRAPH_FIELDS | {'composition_probabilities'}
    out = {k: v for k, v in batch.items() if k in allowed}
    known = out['fragment_mask']
    counts = known.sum(1).view(-1).long()
    maximum = int(counts.max())
    if maximum <= 0:
        raise ValueError('At least one observed atom is required per synthon')
    expected = (torch.arange(known.shape[1], device=known.device)[None, :, None] <
        counts[:, None, None]).to(known)
    if torch.any(counts <= 0) or not torch.equal(known, expected):
        raise ValueError('Expected a nonempty known prefix per synthon')
    for k in ['positions', 'one_hot', 'charges', 'anchors', 'fragment_mask', 'observed_features']:
        if k in out:
            value = out[k][:, :maximum]
            mask = known[:, :maximum]
            while mask.ndim > value.ndim:
                mask = mask.squeeze(-1)
            out[k] = torch.where(mask.bool(), value, torch.zeros_like(value))
    if 'observed_adjacency' in out:
        adjacency = out['observed_adjacency'][:, :maximum, :maximum]
        mask = out['fragment_mask']
        out['observed_adjacency'] = torch.where((mask @ mask.transpose(1, 2)).bool(),
            adjacency, torch.zeros_like(adjacency))
    out['num_atoms'] = [int(n) for n in counts.tolist()]
    out['linker_mask'] = torch.zeros_like(out['fragment_mask'])
    out['atom_mask'] = out['fragment_mask'].clone()
    pair_mask = out['atom_mask'] @ out['atom_mask'].transpose(1, 2)
    diagonal = torch.eye(maximum, device=pair_mask.device, dtype=torch.bool)[None]
    out['edge_mask'] = torch.where(diagonal, torch.zeros_like(pair_mask), pair_mask).reshape(-1, 1)
    return out
