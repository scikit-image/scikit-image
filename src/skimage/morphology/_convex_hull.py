from _skimage2.morphology._convex_hull import possible_hull as possible_hull  # noqa: F401

__all__ = ['possible_hull']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
