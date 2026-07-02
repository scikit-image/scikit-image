from _skimage2.segmentation._clear_border import clear_border as clear_border  # noqa: F401

__all__ = ['clear_border']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
