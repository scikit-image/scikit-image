from _skimage2.filters._sparse import correlate_sparse as correlate_sparse  # noqa: F401

__all__ = ['correlate_sparse']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
