from _skimage2._shared._geometry import (
    polygon_clip as polygon_clip,
    polygon_area as polygon_area,
)  # noqa: F401

__all__ = [
    'polygon_clip',
    'polygon_area',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
