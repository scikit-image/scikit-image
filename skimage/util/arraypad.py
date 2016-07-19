from __future__ import division, absolute_import, print_function

import numpy as np


def pad(array, pad_width, mode, **kwargs):
    return np.pad(array, pad_width, mode, **kwargs)

# Pull function info / docs from NumPy
pad.__doc__ = np.pad.__doc__
