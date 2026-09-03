"""Graph-based operations, e.g., shortest paths.

This includes creating adjacency graphs of pixels in an image, finding the
central pixel in an image, finding (minimum-cost) paths across pixels, merging
and cutting of graphs, etc."""

from _skimage2 import graph as _impl

# `_skimage2.graph` resolves its own attributes lazily; forward to it.
__getattr__ = _impl.__getattr__
__dir__ = _impl.__dir__
__all__ = _impl.__all__
