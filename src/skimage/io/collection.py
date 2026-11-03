"""
Data structures to hold collections of images, with optional caching.
"""

import functools

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

_ski2_imread_collection_wrapper = imread_collection_wrapper


@functools.wraps(_ski2_imread_collection_wrapper)
def imread_collection_wrapper(imread):
    ski2_imread_collection = _ski2_imread_collection_wrapper(imread)

    @functools.wraps(ski2_imread_collection)
    def imread_collection(load_pattern, conserve_memory=True):
        return ImageCollection(
            load_pattern, conserve_memory=conserve_memory, load_func=imread
        )

    return imread_collection


imread_collection_wrapper.__module__ = __name__
