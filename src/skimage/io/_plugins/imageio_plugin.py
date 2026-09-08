from _skimage2.io._plugins.imageio_plugin import (
    imread as imread,
    imsave as imsave,
)  # noqa: F401

__all__ = [
    'imread',
    'imsave',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
