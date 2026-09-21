"""Algorithms to partition images into meaningful regions or boundaries."""

from _skimage2 import segmentation as _impl

# `_skimage2.segmentation` resolves its own attributes lazily; forward to it.
__getattr__ = _impl.__getattr__
__dir__ = _impl.__dir__
__all__ = _impl.__all__
