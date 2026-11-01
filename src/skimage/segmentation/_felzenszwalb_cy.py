from _skimage2.segmentation._felzenszwalb_cy import (
    gaussian as gaussian,
    img_as_float64 as img_as_float64,
    warn as warn,
)  # noqa: F401

__all__ = [
    'gaussian',
    'img_as_float64',
    'warn',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
