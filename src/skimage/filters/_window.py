from _skimage2.filters._window import window as window  # noqa: F401

__all__ = ['window']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
