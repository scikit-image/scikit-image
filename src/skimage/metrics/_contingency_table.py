from _skimage2.metrics._contingency_table import contingency_table as contingency_table  # noqa: F401

__all__ = ['contingency_table']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
