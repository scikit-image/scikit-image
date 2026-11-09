from _skimage2.metrics._variation_of_information import (
    variation_of_information as variation_of_information,
)  # noqa: F401

__all__ = ['variation_of_information']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
