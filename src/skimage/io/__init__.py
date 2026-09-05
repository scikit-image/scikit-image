"""Reading and saving of images and videos."""

from .sift import load_sift, load_surf
from .collection import (
    concatenate_images,
    imread_collection_wrapper,
    ImageCollection,
    MultiImage,
)

from ._io import imread, imread_collection, imsave
from ._image_stack import push, pop


__all__ = [
    "concatenate_images",
    "imread",
    "imread_collection",
    "imread_collection_wrapper",
    "imsave",
    "load_sift",
    "load_surf",
    "pop",
    "push",
    "ImageCollection",
    "MultiImage",
]
