from _skimage2.util._invert import invert as invert  # noqa: F401

__all__ = ['invert']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
