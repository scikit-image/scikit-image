from _skimage2.draw._random_shapes import (
    SHAPE_CHOICES as SHAPE_CHOICES,
    SHAPE_GENERATORS as SHAPE_GENERATORS,
    random_shapes as random_shapes,
)  # noqa: F401

__all__ = [
    'SHAPE_CHOICES',
    'SHAPE_GENERATORS',
    'random_shapes',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
