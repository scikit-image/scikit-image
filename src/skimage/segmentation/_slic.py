from _skimage2.segmentation._slic import regular_grid as regular_grid  # noqa: F401

__all__ = ['regular_grid']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
