_submodules = [
    'color',
    'data',
    'draw',
    'exposure',
    'feature',
    'filters',
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

__all__ = _submodules + ['__version__', 'ExperimentalAPIWarning']  # noqa: F822

from . import (
    color,
    data,
    draw,
    exposure,
    feature,
    filters,
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

# Legacy imports, not advertised in __all__
from .util.dtype import (
    dtype_limits,
    img_as_float32,
    img_as_float64,
    img_as_float,
    img_as_int,
    img_as_uint,
    img_as_ubyte,
    img_as_bool,
)
from .util._lookfor import lookfor
from .data import data_dir

# Used by ``skimage2`` on import; must be a real module attribute (see __init__.py).
from ._shared._warnings import ExperimentalAPIWarning
