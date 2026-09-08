from _skimage2._shared.fast_exp import fast_exp as fast_exp  # noqa: F401

__all__ = ['fast_exp']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
