from _skimage2.transform.pyramids import (
    pyramid_expand as pyramid_expand,
    pyramid_gaussian as pyramid_gaussian,
    pyramid_laplacian as pyramid_laplacian,
    pyramid_reduce as pyramid_reduce,
)  # noqa: F401

__all__ = [
    'pyramid_expand',
    'pyramid_gaussian',
    'pyramid_laplacian',
    'pyramid_reduce',
]

from _skimage2.transform.pyramids import _check_factor  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
