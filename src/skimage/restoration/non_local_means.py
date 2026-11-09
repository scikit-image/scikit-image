from _skimage2.restoration.non_local_means import denoise_nl_means as denoise_nl_means  # noqa: F401

__all__ = ['denoise_nl_means']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
