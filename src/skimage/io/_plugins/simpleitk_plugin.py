from _skimage2.io._plugins.simpleitk_plugin import (
    imread as imread,
    imsave as imsave,
)  # noqa: F401

__all__ = [
    'imread',
    'imsave',
]

from skimage._docutils import adapt_doctests

adapt_doctests(globals())
