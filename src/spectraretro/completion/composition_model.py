"""Select the completion implementation and initialize the shared diffusion backbone."""
from importlib import import_module

from spectraretro.completion.api import setup

setup()
from graph_package import UnknownObservedGraphDDPM, collate_graph, inference_inputs


def implementation(mode):
    if mode not in ('known', 'unknown'):
        raise ValueError('Unsupported class protocol')
    return import_module(f'spectraretro.{mode}-class.completion')


def construct(mode, **config):
    return implementation(mode).construct(**config)


def __getattr__(name):
    # Older checkpoints and scripts may refer to these classes by this module path.
    if name in ('ClassResidual', 'KnownCompositionDDPM'):
        return getattr(implementation('known'), name)
    raise AttributeError(name)
