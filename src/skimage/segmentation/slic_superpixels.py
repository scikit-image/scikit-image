from _skimage2.segmentation.slic_superpixels import slic as slic  # noqa: F401

__all__ = ['slic']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
