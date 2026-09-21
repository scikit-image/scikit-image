"""Drawing primitives, such as lines, circles, text, etc."""

from _skimage2 import draw as _impl

# `_skimage2.draw` resolves its own attributes lazily; forward to it.
__getattr__ = _impl.__getattr__
__dir__ = _impl.__dir__
__all__ = _impl.__all__
