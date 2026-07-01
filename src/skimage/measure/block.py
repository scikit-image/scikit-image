from _skimage2.measure.block import block_reduce as block_reduce  # noqa: F401

__all__ = ['block_reduce']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
