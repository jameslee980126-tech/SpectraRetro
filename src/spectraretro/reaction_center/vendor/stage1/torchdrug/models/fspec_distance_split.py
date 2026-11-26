"""Single-variable distance ablation; not the selected default model."""
import torch
from torch import nn
from .fspec_faithful_rc import FSpecFaithfulReactionCenter


class FSpecDistanceSplit(FSpecFaithfulReactionCenter):
    """Separate exact max_distance, longer finite paths and disconnected pairs.

    New rows start as copies of the old clipped row. This preserves the whole
    initial function and RNG consumption, including downstream task heads.
    Only the subsequent gradients may distinguish the three distance groups.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.distance_embedding.weight.detach()
        expanded = torch.cat((old, old[-1:].repeat(2, 1)), dim=0)
        self.distance_embedding = nn.Embedding.from_pretrained(expanded, freeze=False)

    def _distance_indices(self, graph, device):
        n = int(graph.num_node)
        distance = torch.full((n, n), float('inf'), device=device)
        diagonal = torch.arange(n, device=device)
        distance[diagonal, diagonal] = 0
        if graph.edge_list.numel():
            endpoints = graph.edge_list[:, :2].long()
            distance[endpoints[:, 0], endpoints[:, 1]] = 1
            distance[endpoints[:, 1], endpoints[:, 0]] = 1
        for pivot in range(n):
            distance = torch.minimum(distance, distance[:, pivot, None] + distance[None, pivot, :])
        return torch.where(torch.isfinite(distance),
                           distance.clamp(max=self.max_distance + 1),
                           self.max_distance + 2).long()
