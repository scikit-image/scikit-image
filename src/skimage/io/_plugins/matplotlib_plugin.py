from _skimage2.io._plugins.matplotlib_plugin import (
    ImageProperties as ImageProperties,
    imread as imread,
    imshow as imshow,
    imshow_collection as imshow_collection,
)  # noqa: F401

__all__ = [
    'ImageProperties',
    'imread',
    'imshow',
    'imshow_collection',
]

from skimage._docutils import adapt_doctests

adapt_doctests(globals())
