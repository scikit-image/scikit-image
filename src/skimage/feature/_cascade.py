from _skimage2.feature._cascade import (
    Cascade as Cascade,
    ET as ET,
    integral_image as integral_image,
    math as math,
    rgb2gray as rgb2gray,
)  # noqa: F401

__all__ = [
    'Cascade',
    'ET',
    'integral_image',
    'math',
    'rgb2gray',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
