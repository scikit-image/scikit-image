"""Morphological algorithms, e.g., closing, opening, skeletonization."""

from _skimage2 import morphology as _impl

# `_skimage2.morphology` resolves its own attributes lazily; forward to it.
__getattr__ = _impl.__getattr__
__dir__ = _impl.__dir__
__all__ = _impl.__all__
