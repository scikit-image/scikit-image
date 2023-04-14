__submodules = [
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

__root_level_api = [
    '__version__',
    'dtype_limits',
    'img_as_float32',
    'img_as_float64',
    'img_as_float',
    'img_as_int',
    'img_as_uint',
    'img_as_ubyte',
    'img_as_bool',
    'lookfor',
]

__all__ = __submodules + __root_level_api

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

from .util.dtype import (
    dtype_limits,
    img_as_float32,
    img_as_float64,
    img_as_float,
    img_as_int,
    img_as_uint,
    img_as_ubyte,
    img_as_bool
)

# Legacy import, not advertised in __all__
from .util.lookfor import lookfor
from .data import data_dir
