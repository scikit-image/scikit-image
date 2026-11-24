from _skimage2.feature.sift import SIFT as SIFT  # noqa: F401

__all__ = ['SIFT']

# Make the shim classes for the base classes first.
from . import util  # noqa: F401
from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
