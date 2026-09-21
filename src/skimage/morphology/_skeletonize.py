"""

Algorithms for computing the skeleton of a binary image

"""

from _skimage2.morphology._skeletonize import (
    G123P_LUT as G123P_LUT,
    G123_LUT as G123_LUT,
    medial_axis as medial_axis,
    skeletonize as skeletonize,
    thin as thin,
)  # noqa: F401

__all__ = [
    'G123P_LUT',
    'G123_LUT',
    'medial_axis',
    'skeletonize',
    'thin',
]

from _skimage2.morphology._skeletonize import (  # noqa: F401
    _generate_thin_luts,
)

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
