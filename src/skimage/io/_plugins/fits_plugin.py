from _skimage2.io._plugins.fits_plugin import (
    imread as imread,
    imread_collection as imread_collection,
)  # noqa: F401

__all__ = [
    'imread',
    'imread_collection',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
