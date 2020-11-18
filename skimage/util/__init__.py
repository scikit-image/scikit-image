import functools
import warnings
import numpy as np
from .dtype import (img_as_float32, img_as_float64, img_as_float,
                    img_as_int, img_as_uint, img_as_ubyte,
                    img_as_bool, dtype_limits)
from .shape import view_as_blocks, view_as_windows
from .noise import random_noise
from .apply_parallel import apply_parallel

from .arraycrop import crop
from .compare import compare_images
from ._regular_grid import regular_grid, regular_seeds
from .unique import unique_rows
from ._invert import invert
from ._montage import montage
from ._map_array import map_array


@functools.wraps(np.pad)
def pad(*args, **kwargs):
    warnings.warn("skimage.util.pad is deprecated and will be removed in "
                  "version 0.19. Please use numpy.pad instead.",
                  FutureWarning, stacklevel=2)
    return np.pad(*args, **kwargs)


__all__ = ['img_as_float32',
           'img_as_float64',
           'img_as_float',
           'img_as_int',
           'img_as_uint',
           'img_as_ubyte',
           'img_as_bool',
           'dtype_limits',
           'view_as_blocks',
           'view_as_windows',
           'pad',
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
           ]
