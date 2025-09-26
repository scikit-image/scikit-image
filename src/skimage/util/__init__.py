"""Generic utilities.

This module contains a number of utility functions to work with images in general.
"""

import functools
import warnings

import numpy as np

# keep .dtype imports first to avoid circular imports
from .dtype import (
    dtype_limits,
    img_as_float,
    img_as_float32,
    img_as_float64,
    img_as_bool,
    img_as_int,
    img_as_ubyte,
    img_as_uint,
)
from ._slice_along_axes import slice_along_axes
from ._invert import invert
from ._label import label_points
from ._montage import montage
from ._map_array import map_array
from ._regular_grid import regular_grid, regular_seeds
from .apply_parallel import apply_parallel
from .arraycrop import crop
from .compare import compare_images
from .noise import random_noise
from .shape import view_as_blocks, view_as_windows
from .unique import unique_rows
from .lookfor import lookfor
from .._shared.utils import FailedEstimationAccessError


__all__ = [
    'img_as_float32',
    'img_as_float64',
    'img_as_float',
    'img_as_int',
    'img_as_uint',
    'img_as_ubyte',
    'img_as_bool',
    'dtype_limits',
    'view_as_blocks',
    'view_as_windows',
    'slice_along_axes',
    'crop',
    'compare_images',
    'map_array',
    'montage',
    'random_noise',
    'regular_grid',
    'regular_seeds',
    'apply_parallel',
    'invert',
    'unique_rows',
    'label_points',
    'lookfor',
    'FailedEstimationAccessError',
    'PendingSkimage2Change',
]


class PendingSkimage2Change(PendingDeprecationWarning):
    """A warning about API usage that will silently change or break in skimage2.

    As a subclass of :class:`PendingDeprecationWarning`, this warning isn't
    shown by default. But it can be enabled with a warnings filter to prepare
    for code changes related to skimage2 early on:

    .. code-block:: python

        import warnings
        import skimage as ski
        warnings.filterwarnings(
            action="default", category=ski.util.PendingSkimage2Change
        )
    """
