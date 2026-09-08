"""Image registration algorithms, e.g., optical flow or phase cross correlation."""

from _skimage2 import registration as _impl

# `_skimage2.registration` resolves its own attributes lazily; forward to it.
__getattr__ = _impl.__getattr__
__dir__ = _impl.__dir__
__all__ = _impl.__all__
