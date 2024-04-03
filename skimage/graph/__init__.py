"""
This moddule provides utilities for graph-based image processing.

This includes creating adjacency graphs of pixels in an image, finding the
central pixel in an image, finding (minimum-cost) paths across pixels, merging
and cutting of graphs, etc.

"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
