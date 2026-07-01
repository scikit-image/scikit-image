from _skimage2.restoration.inpaint import inpaint_biharmonic as inpaint_biharmonic  # noqa: F401

__all__ = ['inpaint_biharmonic']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
