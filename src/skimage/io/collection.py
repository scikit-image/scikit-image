"""
Data structures to hold collections of images, with optional caching.
"""

from _skimage2.io.collection import (
    MultiImage as MultiImage,
    ImageCollection as ImageCollection,
    concatenate_images as concatenate_images,
    imread_collection_wrapper as imread_collection_wrapper,
)  # noqa: F401

__all__ = [
    'MultiImage',
    'ImageCollection',
    'concatenate_images',
    'imread_collection_wrapper',
]

from _skimage2.io.collection import alphanumeric_key  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
