from __future__ import division, absolute_import, print_function

from numpy import pad as numpy_pad


def pad(array, pad_width, mode, **kwargs):
    return numpy_pad(array, pad_width, mode, **kwargs)

# Pull function info / docs from NumPy
pad.__doc__ = numpy_pad.__doc__
