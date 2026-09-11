"""Rank filters, e.g. median, entropy, local Otsu, etc."""

from _skimage2.filters import rank as _impl

# `_skimage2.filters.rank` resolves its own attributes lazily; forward to it.
__getattr__ = _impl.__getattr__
__dir__ = _impl.__dir__
__all__ = _impl.__all__
