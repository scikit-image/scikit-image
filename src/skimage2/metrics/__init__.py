"""Metrics corresponding to images, e.g., distance metrics, similarity, etc."""

from _skimage2 import metrics as _impl

# `_skimage2.metrics` resolves its own attributes lazily; forward to it.
__getattr__ = _impl.__getattr__
__dir__ = _impl.__dir__
__all__ = _impl.__all__
