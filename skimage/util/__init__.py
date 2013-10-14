from .dtype import (img_as_float, img_as_int, img_as_uint, img_as_ubyte,
                    img_as_bool, dtype_limits)
from .shape import view_as_blocks, view_as_windows
from .noise import random_noise

import numpy
ver = numpy.__version__.split('.')
chk = int(ver[0] + ver[1])
if chk < 18:    # Use internal version for numpy versions < 1.8.x
    from .arraypad import pad
else:
    from numpy import pad
del numpy, ver, chk
from ._regular_grid import regular_grid
from .unique import unique_rows


__all__ = ['img_as_float',
           'img_as_int',
           'img_as_uint',
           'img_as_ubyte',
           'img_as_bool',
           'dtype_limits',
           'view_as_blocks',
           'view_as_windows',
           'pad',
           'random_noise',
           'regular_grid',
           'unique_rows']
