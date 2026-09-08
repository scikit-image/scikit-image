"""

:author: Stefan van der Walt, 2008
:license: modified BSD

"""

from _skimage2.filters.lpi_filter import (
    LPIFilter2D as LPIFilter2D,
    filter_forward as filter_forward,
    filter_inverse as filter_inverse,
    wiener as wiener,
)  # noqa: F401

__all__ = [
    'LPIFilter2D',
    'filter_forward',
    'filter_inverse',
    'wiener',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
