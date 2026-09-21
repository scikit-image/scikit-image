from _skimage2.segmentation._chan_vese import chan_vese as chan_vese  # noqa: F401

__all__ = ['chan_vese']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
