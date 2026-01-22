"""Inferior and superior ranks, provided by the user, are passed to the kernel
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

from ..._shared.utils import check_nD
from . import percentile_cy
from .generic import _preprocess_input

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


def _apply(func, image, footprint, out, mask, shift_x, shift_y, p0, p1, out_dtype=None):
    check_nD(image, 2)
    image, footprint, out, mask, n_bins = _preprocess_input(
        image,
        footprint,
        out,
        mask,
        out_dtype,
        shift_x=shift_x,
        shift_y=shift_y,
    )

    func(
        image,
        footprint,
        shift_x=shift_x,
        shift_y=shift_y,
        mask=mask,
        out=out,
        n_bins=n_bins,
        p0=p0,
        p1=p1,
    )

    return out.reshape(out.shape[:2])


def autolevel_percentile(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
):
    """Return grayscale local autolevel of an image.

    This filter locally stretches the histogram of grayvalues to cover the
    entire range of values from "white" to "black".

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : ndarray of shape (M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray of shape (m, n)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray of shape (M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0, p1 : float, optional, in interval [0, 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    out : ndarray of shape (M, N), same dtype as input `image`
        Output image.

    """

    return _apply(
        percentile_cy._autolevel,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
    )


def gradient_percentile(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
):
    """Return local gradient of an image (i.e., local maximum - local minimum).

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : ndarray of shape (M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray of shape (m, n)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray of shape (M, N) and dtype int
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0, p1 : float, optional, in interval [0, 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    out : ndarray of shape (M, N) and dtype int
        Output image.

    """

    return _apply(
        percentile_cy._gradient,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
    )


def mean_percentile(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
):
    """Return local mean of an image.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : ndarray of shape (M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray of shape (m, n)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray of shape (M, N) and dtype int
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0, p1 : float, optional, in interval [0, 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    out : ndarray of shape (M, N) and dtype int
        Output image.

    """

    return _apply(
        percentile_cy._mean,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
    )


def subtract_mean_percentile(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
):
    """
    Subtract the local mean (computed within the specified percentile window) from each pixel.
    
    Only grayvalues between percentiles [p0, p1] are considered when computing the local mean.
    
    Parameters:
        image (ndarray of shape (M, N) and dtype (uint8 or uint16)):
            Input image.
        footprint (ndarray of shape (m, n)):
            Neighborhood as a 2-D array of 1's and 0's.
        out (ndarray of shape (M, N), same dtype as input `image`, optional):
            Array to store the result. If None, a new array is allocated.
        mask (ndarray, optional):
            Mask (>0) defining pixels included in local neighborhoods. If None, the full image is used.
        shift_x (int), shift_y (int), optional:
            Offsets added to the footprint center; shifts are bounded so the center remains inside the footprint.
        p0 (float, optional):
            Lower percentile in [0, 1] for the percentile window.
        p1 (float, optional):
            Upper percentile in [0, 1] for the percentile window.
    
    Returns:
        out (ndarray of shape (M, N), same dtype as input `image`):
            Image with the local mean (within [p0, p1]) subtracted from each pixel.
    """

    return _apply(
        percentile_cy._subtract_mean,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
    )


def enhance_contrast_percentile(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
):
    """Enhance contrast of an image.

    This replaces each pixel by the local maximum if the pixel grayvalue is
    closer to the local maximum than the local minimum. Otherwise it is
    replaced by the local minimum.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : ndarray of shape (M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray of shape (m, n)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray of shape (M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0, p1 : float, optional, in interval [0, 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    out : ndarray of shape (M, N), same dtype as input `image`
        Output image.

    """

    return _apply(
        percentile_cy._enhance_contrast,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
    )


def percentile(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0):
    """
    Compute the local p0 lower percentile value for each pixel.
    
    The filter computes, for every image location, the p0 lower percentile of grayvalues
    within the neighborhood defined by `footprint`. Only grayvalues within the percentile
    window [p0, 1) are considered.
    
    Parameters:
        image (ndarray of shape (M, N) and dtype (uint8 or uint16)):
            Input image.
        footprint (ndarray of shape (m, n)):
            Neighborhood expressed as a 2-D array of 1's and 0's.
        out (ndarray of shape (M, N), same dtype as input `image`, optional):
            Destination array to store the result. If None, a new array is allocated.
        mask (ndarray, optional):
            Mask that defines (>0) pixels included in the local neighborhood. If None,
            the entire image is used.
        shift_x (int, optional):
            Horizontal offset added to the footprint center; bounded to footprint size.
        shift_y (int, optional):
            Vertical offset added to the footprint center; bounded to footprint size.
        p0 (float, optional, in interval [0, 1]):
            Lower percentile to compute.
    
    Returns:
        out (ndarray of shape (M, N), same dtype as input `image`):
            Array of local p0 lower percentile values for each pixel.
    """

    return _apply(
        percentile_cy._percentile,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=0.0,
    )


def pop_percentile(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
):
    """
    Compute the local count of pixels within the footprint and optional mask whose values lie between the p0 and p1 percentiles.
    
    Only pixels included by the footprint and (if provided) the mask are considered; among those, only grayvalues within the percentile interval [p0, p1] contribute to the count.
    
    Parameters:
        image (ndarray of shape (M, N) and dtype (uint8 or uint16)):
            Input image.
        footprint (ndarray of shape (m, n)):
            Neighborhood expressed as a 2-D array of 1's and 0's.
        out (ndarray of shape (M, N), same dtype as input `image`, optional):
            Destination array for the result. If None, a new array is allocated.
        mask (ndarray, optional):
            Mask array that defines (>0) area of the image included in the local neighborhood. If None, the complete image is used.
        shift_x (int, optional):
            Horizontal offset added to the footprint center; bounded so the center stays inside the footprint.
        shift_y (int, optional):
            Vertical offset added to the footprint center; bounded so the center stays inside the footprint.
        p0, p1 (float, optional, in interval [0, 1]):
            Lower and upper percentile bounds defining the inclusive interval of grayvalues considered.
    
    Returns:
        out (ndarray of shape (M, N), same dtype as input `image`):
            Per-pixel local population count of values within the specified percentile interval.
    """

    return _apply(
        percentile_cy._pop,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
    )


def sum_percentile(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0, p1=1
):
    """
    Compute the local sum of pixels within the given percentile interval.
    
    Only grayvalues between percentiles [p0, p1] are considered in the filter.
    Note that the sum may overflow depending on the dtype of the input array.
    
    Parameters
    ----------
    image : ndarray of shape (M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray of shape (m, n)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray of shape (M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray, optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0, p1 : float, optional, in interval [0, 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.
    
    Returns
    -------
    out : ndarray of shape (M, N), same dtype as input `image`
        Output image containing the local sums.
    """

    return _apply(
        percentile_cy._sum,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
    )


def threshold_percentile(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, p0=0
):
    """
    Compute a local binary mask indicating whether each center pixel exceeds the local mean
    computed within a percentile-restricted neighborhood.
    
    Parameters
    ----------
    image : ndarray of shape (M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray of shape (m, n)
        Neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray of shape (M, N), dtype bool, optional
        Destination array for the result. If None, a new boolean array is allocated.
    mask : ndarray, optional
        Mask array that defines (>0) area of the image included in the local neighborhood.
        If None, the complete image is used.
    shift_x, shift_y : int, optional
        Offset added to the footprint center point; bounded to footprint sizes.
    p0 : float, optional, in interval [0, 1]
        Lower percentile defining the percentile window [p0, 1] used to restrict values
        considered when computing the local mean.
    
    Returns
    -------
    out : ndarray of shape (M, N), dtype bool
        Local binary mask: `True` if the center pixel value is greater than the local mean
        computed over the neighborhood considering only pixels within percentiles [p0, 1],
        `False` otherwise.
    """

    return _apply(
        percentile_cy._threshold,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=0,
    )