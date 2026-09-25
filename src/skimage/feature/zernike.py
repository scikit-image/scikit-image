from _skimage2.feature.zernike import zernike_features as zernike_features  # noqa: F401

__all__ = ['zernike_features']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
