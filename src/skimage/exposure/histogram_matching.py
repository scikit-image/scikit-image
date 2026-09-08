from _skimage2.exposure.histogram_matching import match_histograms as match_histograms  # noqa: F401

__all__ = ['match_histograms']

from _skimage2.exposure.histogram_matching import _match_cumulative_cdf  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
