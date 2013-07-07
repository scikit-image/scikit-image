"""Inferior and superior ranks, provided by the user, are passed to the kernel
function to provide a softer version of the rank filters. E.g.
percentile_autolevel will stretch image levels between percentile [p0, p1]
instead of using [min, max]. It means that isolated bright or dark pixels will
not produce halos.

The local histogram is computed using a sliding window similar to the method
described in [1]_.

Input image can be 8-bit or 16-bit, for 16-bit input images, the number of
histogram bins is determined from the maximum value present in the image.

Result image is 8 or 16-bit with respect to the input image.

References
----------

.. [1] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

"""

import numpy as np

from . import percentile_cy
from .generic import _handle_input


__all__ = ['percentile_autolevel', 'percentile_gradient',
           'percentile_mean', 'percentile_subtract_mean',
           'percentile_enhance_contrast', 'percentile', 'percentile_pop',
           'percentile_threshold']


def _apply(func, image, selem, out, mask, shift_x, shift_y, p0, p1):

    image, selem, out, mask, max_bin = _handle_input(image, selem, out, mask)

    func(image, selem, shift_x=shift_x, shift_y=shift_y, mask=mask,
         out=out, max_bin=max_bin, p0=p0, p1=p1)

    return out


def percentile_autolevel(image, selem, out=None, mask=None, shift_x=False,
                         shift_y=False, p0=0, p1=1):
    """Return greyscale local autolevel of an image.

    Autolevel is computed on the given structuring element. Only levels between
    percentiles [p0, p1] are used.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    local autolevel : ndarray (same dtype as input)
        The result of the local autolevel.

    """

    return _apply(percentile_cy._autolevel,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


def percentile_gradient(image, selem, out=None, mask=None, shift_x=False,
                        shift_y=False, p0=0, p1=1):
    """Return greyscale local percentile_gradient of an image.

    percentile_gradient is computed on the given structuring element. Only
    levels between percentiles [p0, p1] are used.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    local percentile_gradient : ndarray (same dtype as input)
        The result of the local percentile_gradient.

    """

    return _apply(percentile_cy._gradient,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


def percentile_mean(image, selem, out=None, mask=None, shift_x=False,
                    shift_y=False, p0=0, p1=1):
    """Return greyscale local mean of an image.

    Mean is computed on the given structuring element. Only levels between
    percentiles [p0, p1] are used.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    local mean : ndarray (same dtype as input)
        The result of the local mean.

    """

    return _apply(percentile_cy._mean,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


def percentile_subtract_mean(image, selem, out=None, mask=None,
                                 shift_x=False, shift_y=False, p0=0, p1=1):
    """Return greyscale local subtract_mean of an image.

    subtract_mean is computed on the given structuring element. Only levels
    between percentiles [p0, p1] are used.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    local subtract_mean : ndarray (same dtype as input)
        The result of the local subtract_mean.

    """

    return _apply(percentile_cy._subtract_mean,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


def percentile_enhance_contrast(image, selem, out=None, mask=None,
                               shift_x=False, shift_y=False, p0=0, p1=1):
    """Return greyscale local enhance_contrast of an image.

    enhance_contrast is computed on the given structuring element. Only levels
    between percentiles [p0, p1] are used.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    local enhance_contrast : ndarray (same dtype as input)
        The result of the local enhance_contrast.

    """

    return _apply(percentile_cy._enhance_contrast,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


def percentile(image, selem, out=None, mask=None, shift_x=False, shift_y=False,
               p0=0):
    """Return greyscale local percentile of an image.

    percentile is computed on the given structuring element. Returns the value
    of the p0 lower percentile of the neighborhood value distribution.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    p0 : float in [0, ..., 1]
        Set the percentile value.

    Returns
    -------
    local percentile : ndarray (same dtype as input)
        The result of the local percentile.

    """

    return _apply(percentile_cy._percentile,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=0.)


def percentile_pop(image, selem, out=None, mask=None, shift_x=False,
                   shift_y=False, p0=0, p1=1):
    """Return greyscale local pop of an image.

    pop is computed on the given structuring element. Only levels between
    percentiles [p0, p1] are used.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    local pop : ndarray (same dtype as input)
        The result of the local pop.

    """

    return _apply(percentile_cy._pop,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


def percentile_threshold(image, selem, out=None, mask=None, shift_x=False,
                         shift_y=False, p0=0):
    """Return greyscale local threshold of an image.

    threshold is computed on the given structuring element. Returns
    thresholded image such that pixels having a higher value than the the p0
    percentile of the neighborhood value distribution are set to 2^nbit-1
    (e.g. 255 for 8bit image).

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    p0 : float in [0, ..., 1]
        Set the percentile value.

    Returns
    -------
    local threshold : ndarray (same dtype as input)
        The result of the local threshold.

    """

    return _apply(percentile_cy._threshold,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=0)
