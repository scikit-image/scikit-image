"""Measurement of image properties, e.g., region properties, contours."""

from _skimage2 import measure as _impl

# `_skimage2.measure` resolves its own attributes lazily; forward to it.
__getattr__ = _impl.__getattr__
__dir__ = _impl.__dir__
__all__ = _impl.__all__
