from _skimage2.draw.draw_nd import line_nd as line_nd  # noqa: F401

__all__ = ['line_nd']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
