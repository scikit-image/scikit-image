"""
extrema.py - local minima and maxima

This module provides functions to find local maxima and minima of an image.
Here, local maxima (minima) are defined as connected sets of pixels with equal
gray level which is strictly greater (smaller) than the gray level of all
pixels in direct neighborhood of the connected set. In addition, the module
provides the related functions h-maxima and h-minima.

Soille, P. (2003). Morphological Image Analysis: Principles and Applications
(2nd ed.), Chapter 6. Springer-Verlag New York, Inc.

"""

from _skimage2.morphology.extrema import (
    h_maxima as h_maxima,
    h_minima as h_minima,
    local_maxima as local_maxima,
    local_minima as local_minima,
)  # noqa: F401

__all__ = [
    'h_maxima',
    'h_minima',
    'local_maxima',
    'local_minima',
]

from _skimage2.morphology.extrema import (  # noqa: F401
    _add_constant_clip,
    _subtract_constant_clip,
)

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
