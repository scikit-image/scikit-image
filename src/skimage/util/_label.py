from _skimage2.util._label import label_points as label_points  # noqa: F401

__all__ = ['label_points']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
