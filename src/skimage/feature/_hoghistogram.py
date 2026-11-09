from _skimage2.feature._hoghistogram import hog_histograms as hog_histograms  # noqa: F401

__all__ = ['hog_histograms']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
