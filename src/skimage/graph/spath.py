from _skimage2.graph.spath import shortest_path as shortest_path  # noqa: F401

__all__ = ['shortest_path']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
