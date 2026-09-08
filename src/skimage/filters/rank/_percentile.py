"""
Inferior and superior ranks, provided by the user, are passed to the kernel
function to provide a softer version of the rank filters. E.g.
``autolevel_percentile`` will stretch image levels between percentile [p0, p1]
instead of using [min, max]. It means that isolated bright or dark pixels will
not produce halos.

The local histogram is computed using a sliding window similar to the method
described in [1]_.

Input image can be 8-bit or 16-bit, for 16-bit input images, the number of
histogram bins is determined from the maximum value present in the image.

Result image is 8-/16-bit or double with respect to the input image and the
rank filter operation.

References
----------

.. [1] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.


"""

from _skimage2.filters.rank._percentile import (
    autolevel_percentile as autolevel_percentile,
    gradient_percentile as gradient_percentile,
    mean_percentile as mean_percentile,
    subtract_mean_percentile as subtract_mean_percentile,
    enhance_contrast_percentile as enhance_contrast_percentile,
    percentile as percentile,
    pop_percentile as pop_percentile,
    threshold_percentile as threshold_percentile,
)  # noqa: F401

__all__ = [
    'autolevel_percentile',
    'gradient_percentile',
    'mean_percentile',
    'subtract_mean_percentile',
    'enhance_contrast_percentile',
    'percentile',
    'pop_percentile',
    'threshold_percentile',
]

from _skimage2.filters.rank._percentile import sum_percentile  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
