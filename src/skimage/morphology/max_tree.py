"""
_max_tree.py - max_tree representation of images.

This module provides operators based on the max-tree representation of images.
A grayscale image can be seen as a pile of nested sets, each of which is the
result of a threshold operation. These sets can be efficiently represented by
max-trees, where the inclusion relation between connected components at
different levels are represented by parent-child relationships.

These representations allow efficient implementations of many algorithms, such
as attribute operators. Unlike morphological openings and closings, these
operators do not require a fixed footprint, but rather act with a flexible
footprint that meets a certain criterion.

This implementation provides functions for:
1. max-tree generation
2. area openings / closings
3. diameter openings / closings
4. local maxima

References:
    .. [1] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
           Connected Operators for Image and Sequence Processing.
           IEEE Transactions on Image Processing, 7(4), 555-570.
           :DOI:`10.1109/83.663500`
    .. [2] Berger, C., Geraud, T., Levillain, R., Widynski, N., Baillard, A.,
           Bertin, E. (2007). Effective Component Tree Computation with
           Application to Pattern Recognition in Astronomical Imaging.
           In International Conference on Image Processing (ICIP) (pp. 41-44).
           :DOI:`10.1109/ICIP.2007.4379949`
    .. [3] Najman, L., & Couprie, M. (2006). Building the component tree in
           quasi-linear time. IEEE Transactions on Image Processing, 15(11),
           3531-3539.
           :DOI:`10.1109/TIP.2006.877518`
    .. [4] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.
           :DOI:`10.1109/TIP.2014.2336551`

"""

from _skimage2.morphology._max_tree import (
    area_closing as area_closing,
    area_opening as area_opening,
    diameter_closing as diameter_closing,
    diameter_opening as diameter_opening,
    max_tree as max_tree,
    max_tree_local_maxima as max_tree_local_maxima,
    signed_float_types as signed_float_types,
    signed_int_types as signed_int_types,
    unsigned_int_types as unsigned_int_types,
)  # noqa: F401

__all__ = [
    'area_closing',
    'area_opening',
    'diameter_closing',
    'diameter_opening',
    'max_tree',
    'max_tree_local_maxima',
    'signed_float_types',
    'signed_int_types',
    'unsigned_int_types',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
