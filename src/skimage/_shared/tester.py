from _skimage2._shared.tester import PytestTester as PytestTester  # noqa: F401

__all__ = ['PytestTester']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
