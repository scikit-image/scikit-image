from _skimage2.io._plugins.pil_plugin import (
    imread as imread,
    imsave as imsave,
)  # noqa: F401

__all__ = [
    'imread',
    'imsave',
]

from _skimage2.io._plugins.pil_plugin import (  # noqa: F401
    _palette_is_grayscale,
    ndarray_to_pil,
    pil_to_ndarray,
)

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
