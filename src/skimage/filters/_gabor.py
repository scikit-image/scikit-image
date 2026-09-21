from _skimage2.filters._gabor import (
    gabor_kernel as gabor_kernel,
    gabor as gabor,
)  # noqa: F401

__all__ = [
    'gabor_kernel',
    'gabor',
]

from _skimage2.filters._gabor import _sigma_prefactor  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
