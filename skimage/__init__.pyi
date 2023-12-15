submodules = [
    'color',
    'data',
    'draw',
    'exposure',
    'feature',
    'filters',
    'future',
    'graph',
    'io',
    'measure',
    'metrics',
    'morphology',
    'registration',
    'restoration',
    'segmentation',
    'transform',
    'util',
]

__all__ = submodules + ['__version__']  # noqa: F822

from . import (
    color,
    data,
    draw,
    exposure,
    feature,
    filters,
    future,
    graph,
    io,
    measure,
    metrics,
    morphology,
    registration,
    restoration,
    segmentation,
    transform,
    util,
)

__version__ = ""
