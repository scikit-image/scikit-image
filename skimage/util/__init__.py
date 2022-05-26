import functools
import warnings

import numpy as np

from ._multimethods import (compare_images, crop, dtype_limits, img_as_bool,
                            img_as_float, img_as_float32, img_as_float64,
                            img_as_int, img_as_ubyte, img_as_uint, invert,
                            label_points, map_array, montage, random_noise,
                            regular_grid, regular_seeds, unique_rows,
                            view_as_blocks, view_as_windows)
from .apply_parallel import apply_parallel

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
           ]
