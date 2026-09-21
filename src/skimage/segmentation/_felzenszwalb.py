from _skimage2.segmentation._felzenszwalb import felzenszwalb as felzenszwalb  # noqa: F401

__all__ = ['felzenszwalb']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
