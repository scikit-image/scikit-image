"""Sharpening, edge finding, rank filters, thresholding, etc."""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)

from ._high_boost import high_boost_filter as high_boost_filter  # noqa: E402, F401
