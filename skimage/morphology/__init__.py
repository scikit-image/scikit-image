"""Utilities that operate on shapes in images.

These operations are particularly suited for binary images,
although some may be useful for images of other types as well.

Basic morphological operations include dilation and erosion.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
