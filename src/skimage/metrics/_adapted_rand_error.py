from _skimage2.metrics._adapted_rand_error import (
    adapted_rand_error as adapted_rand_error,
)  # noqa: F401

__all__ = ['adapted_rand_error']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
