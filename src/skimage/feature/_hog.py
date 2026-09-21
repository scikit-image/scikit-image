from _skimage2.feature._hog import hog as hog  # noqa: F401

__all__ = ['hog']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
