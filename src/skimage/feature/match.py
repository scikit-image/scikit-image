from _skimage2.feature.match import match_descriptors as match_descriptors  # noqa: F401

__all__ = ['match_descriptors']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
