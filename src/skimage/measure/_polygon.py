from _skimage2.measure._polygon import (
    approximate_polygon as approximate_polygon,
    subdivide_polygon as subdivide_polygon,
)  # noqa: F401

__all__ = [
    'approximate_polygon',
    'subdivide_polygon',
]

from _skimage2.measure._polygon import _SUBDIVISION_MASKS  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
