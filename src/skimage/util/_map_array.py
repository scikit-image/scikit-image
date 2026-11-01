from _skimage2.util._map_array import (
    ArrayMap as ArrayMap,
    map_array as map_array,
)  # noqa: F401

__all__ = [
    'ArrayMap',
    'map_array',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
