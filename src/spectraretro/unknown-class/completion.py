"""Class-free 3D synthon completion with a composition prior."""
from spectraretro.completion.composition_model import (
    UnknownObservedGraphDDPM,
    collate_graph as prepare_batch,
    inference_inputs as sampling_inputs,
)


def construct(**config):
    return UnknownObservedGraphDDPM(
        observed_graph_enabled=False, compatibility_enabled=True, **config
    )
