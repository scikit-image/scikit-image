"""

Binary morphological operations

"""

from _skimage2.morphology.binary import (
    binary_closing as binary_closing,
    binary_dilation as binary_dilation,
    binary_erosion as binary_erosion,
    binary_opening as binary_opening,
)  # noqa: F401

__all__ = [
    'binary_closing',
    'binary_dilation',
    'binary_erosion',
    'binary_opening',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
