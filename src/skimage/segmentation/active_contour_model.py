from _skimage2.segmentation.active_contour_model import active_contour as active_contour  # noqa: F401

__all__ = ['active_contour']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
