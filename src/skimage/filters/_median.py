from _skimage2.filters._median import median as median  # noqa: F401

__all__ = ['median']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
