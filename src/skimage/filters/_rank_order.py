"""

_rank_order.py - convert an image of any type to an image of ints whose
pixels have an identical rank order compared to the original image

"""

from _skimage2.filters._rank_order import rank_order as rank_order  # noqa: F401

__all__ = ['rank_order']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
