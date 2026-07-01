from _skimage2.feature._basic_features import (
    multiscale_basic_features as multiscale_basic_features,
)  # noqa: F401

__all__ = ['multiscale_basic_features']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
