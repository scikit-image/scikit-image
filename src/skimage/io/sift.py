from _skimage2.io.sift import (
    load_sift as load_sift,
    load_surf as load_surf,
)  # noqa: F401

__all__ = [
    'load_sift',
    'load_surf',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
