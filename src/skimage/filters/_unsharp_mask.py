from _skimage2.filters._unsharp_mask import unsharp_mask as unsharp_mask  # noqa: F401

__all__ = ['unsharp_mask']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
