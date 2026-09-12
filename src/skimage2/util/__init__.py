"""Utility functions to work with images in general."""

from _skimage2 import util as _impl

# `_skimage2.util` resolves its own attributes lazily; forward to it.
__getattr__ = _impl.__getattr__
__dir__ = _impl.__dir__
__all__ = _impl.__all__
