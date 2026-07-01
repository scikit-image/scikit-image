from _skimage2.measure._moments_cy import moments_hu as moments_hu  # noqa: F401

__all__ = ['moments_hu']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
