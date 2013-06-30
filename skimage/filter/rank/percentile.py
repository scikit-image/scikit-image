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
from skimage import img_as_ubyte
from . import percentile8_cy, percentile16_cy
from .generic import find_bitdepth


__all__ = ['percentile_autolevel', 'percentile_gradient',
           'percentile_mean', 'percentile_mean_subtraction',
           'percentile_morph_contr_enh', 'percentile', 'percentile_pop',
           'percentile_threshold']


def _apply(func8, func16, image, selem, out, mask, shift_x, shift_y, p0, p1):
    selem = img_as_ubyte(selem > 0)
    image = np.ascontiguousarray(image)

    if mask is None:
        mask = np.ones(image.shape, dtype=np.uint8)
    else:
        mask = np.ascontiguousarray(mask)
        mask = img_as_ubyte(mask)

    if image is out:
        raise NotImplementedError("Cannot perform rank operation in place.")

    if image.dtype == np.uint8:
        if func8 is None:
            raise TypeError("Not implemented for uint8 image.")
        if out is None:
            out = np.zeros(image.shape, dtype=np.uint8)
        func8(image, selem, shift_x=shift_x, shift_y=shift_y,
              mask=mask, out=out, p0=p0, p1=p1)
    elif image.dtype == np.uint16:
        if func16 is None:
            raise TypeError("Not implemented for uint16 image.")
        if out is None:
            out = np.zeros(image.shape, dtype=np.uint16)
        bitdepth = find_bitdepth(image)
        func16(image, selem, shift_x=shift_x, shift_y=shift_y, mask=mask,
               bitdepth=bitdepth + 1, out=out, p0=p0, p1=p1)
    else:
        raise TypeError("Only uint8 and uint16 image supported.")

    return out


def percentile_autolevel(image, selem, out=None, mask=None, shift_x=False,
                         shift_y=False, p0=.0, p1=1.):
    """Return greyscale local autolevel of an image.

    Autolevel is computed on the given structuring element. Only levels between
    percentiles [p0, p1] are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
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
    local autolevel : uint8 array or uint16
        The result of the local autolevel.

    """

    return _apply(percentile8_cy.autolevel, percentile16_cy.autolevel,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


def percentile_gradient(image, selem, out=None, mask=None, shift_x=False,
                        shift_y=False, p0=.0, p1=1.):
    """Return greyscale local percentile_gradient of an image.

    percentile_gradient is computed on the given structuring element. Only
    levels between percentiles [p0, p1] are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
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
    local percentile_gradient : uint8 array or uint16
        The result of the local percentile_gradient.

    """

    return _apply(percentile8_cy.gradient, percentile16_cy.gradient,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


def percentile_mean(image, selem, out=None, mask=None, shift_x=False,
                    shift_y=False, p0=.0, p1=1.):
    """Return greyscale local mean of an image.

    Mean is computed on the given structuring element. Only levels between
    percentiles [p0, p1] are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
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
    local mean : uint8 array or uint16
        The result of the local mean.

    """

    return _apply(percentile8_cy.mean, percentile16_cy.mean,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


def percentile_mean_subtraction(image, selem, out=None, mask=None,
                                 shift_x=False, shift_y=False, p0=.0, p1=1.):
    """Return greyscale local mean_subtraction of an image.

    mean_subtraction is computed on the given structuring element. Only levels
    between percentiles [p0, p1] are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
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
    local mean_subtraction : uint8 array or uint16
        The result of the local mean_subtraction.

    """

    return _apply(percentile8_cy.mean_subtraction,
                  percentile16_cy.mean_subtraction,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


def percentile_morph_contr_enh(image, selem, out=None, mask=None,
                               shift_x=False, shift_y=False, p0=.0, p1=1.):
    """Return greyscale local morph_contr_enh of an image.

    morph_contr_enh is computed on the given structuring element. Only levels
    between percentiles [p0, p1] are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
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
    local morph_contr_enh : uint8 array or uint16
        The result of the local morph_contr_enh.

    """

    return _apply(percentile8_cy.morph_contr_enh,
                  percentile16_cy.morph_contr_enh,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


def percentile(image, selem, out=None, mask=None, shift_x=False, shift_y=False,
               p0=.0):
    """Return greyscale local percentile of an image.

    percentile is computed on the given structuring element. Returns the value
    of the p0 lower percentile of the neighborhood value distribution.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
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
    local percentile : uint8 array or uint16
        The result of the local percentile.

    """

    return _apply(percentile8_cy.percentile,
                  percentile16_cy.percentile,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=0.)


def percentile_pop(image, selem, out=None, mask=None, shift_x=False,
                   shift_y=False, p0=.0, p1=1.):
    """Return greyscale local pop of an image.

    pop is computed on the given structuring element. Only levels between
    percentiles [p0, p1] are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
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
    local pop : uint8 array or uint16
        The result of the local pop.

    """

    return _apply(percentile8_cy.pop, percentile16_cy.pop,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


def percentile_threshold(image, selem, out=None, mask=None, shift_x=False,
                         shift_y=False, p0=.0):
    """Return greyscale local threshold of an image.

    threshold is computed on the given structuring element. Returns
    thresholded image such that pixels having a higher value than the the p0
    percentile of the neighborhood value distribution are set to 2^nbit-1
    (e.g. 255 for 8bit image).

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
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
    local threshold : uint8 array or uint16
        The result of the local threshold.

    """

    return _apply(percentile8_cy.threshold, percentile16_cy.threshold,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=0.)
