"""
Implementation of various restoration functions.
"""

from _skimage2.restoration.deconvolution import (
    richardson_lucy as richardson_lucy,
    unsupervised_wiener as unsupervised_wiener,
    wiener as wiener,
)  # noqa: F401

__all__ = [
    'richardson_lucy',
    'unsupervised_wiener',
    'wiener',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
