from _skimage2.segmentation.boundaries import (
    find_boundaries as find_boundaries,
    mark_boundaries as mark_boundaries,
)  # noqa: F401

__all__ = [
    'find_boundaries',
    'mark_boundaries',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
