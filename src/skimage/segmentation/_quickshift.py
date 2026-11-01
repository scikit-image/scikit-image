from _skimage2.segmentation._quickshift import quickshift as quickshift  # noqa: F401

__all__ = ['quickshift']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
