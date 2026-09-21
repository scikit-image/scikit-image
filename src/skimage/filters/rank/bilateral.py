"""
Approximate bilateral rank filter for local (custom kernel) mean.

The local histogram is computed using a sliding window similar to the method
described in [1]_.

The pixel neighborhood is defined by:

* the given footprint (structuring element)
* an interval [g-s0, g+s1] in graylevel around g the processed pixel graylevel

The kernel is flat (i.e. each pixel belonging to the neighborhood contributes
equally).

Result image is 8-/16-bit or double with respect to the input image and the
rank filter operation.

References
----------

.. [1] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.


"""

from _skimage2.filters.rank.bilateral import (
    mean_bilateral as mean_bilateral,
    pop_bilateral as pop_bilateral,
    sum_bilateral as sum_bilateral,
)  # noqa: F401

__all__ = [
    'mean_bilateral',
    'pop_bilateral',
    'sum_bilateral',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
