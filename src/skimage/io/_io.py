from _skimage2.io._io import (
    imread as imread,
    imsave as imsave,
    imread_collection as imread_collection,
)  # noqa: F401

__all__ = [
    'imread',
    'imsave',
    'imread_collection',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
