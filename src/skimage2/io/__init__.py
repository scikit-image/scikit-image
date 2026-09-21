"""Reading and saving of images and videos."""

from _skimage2 import io as _impl

# `_skimage2.io` resolves its own attributes lazily; forward to it.
__getattr__ = _impl.__getattr__
__dir__ = _impl.__dir__
__all__ = _impl.__all__
