"""Approximate bilateral rank filter for local (custom kernel) mean.

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

from ..._shared.utils import check_nD
from . import bilateral_cy
from .generic import _preprocess_input

__all__ = ['mean_bilateral', 'pop_bilateral', 'sum_bilateral']


def _apply(func, image, footprint, out, mask, shift_x, shift_y, s0, s1, out_dtype=None):
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
        s0=s0,
        s1=s1,
    )

    return out.reshape(out.shape[:2])


def mean_bilateral(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10
):
    """
    Compute a local, edge-preserving mean using a flat bilateral filter.
    
    The filter averages pixels inside a spatial neighborhood defined by `footprint`
    and a radiometric interval [g - s0, g + s1] around each center pixel value g.
    Only pixels that belong to the footprint, are inside the optional `mask`, and
    have graylevel within the interval contribute to the mean. The kernel is flat
    (equal weight) and the output dtype follows the input image dtype.
    
    Parameters
    ----------
    image : ndarray of shape (M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray of shape (m, n)
        Neighborhood defined as a 2-D array of 1s and 0s.
    out : ndarray of shape (M, N), same dtype as input `image`, optional
        Array to store the result. If None, a new array is allocated.
    mask : ndarray, optional
        Mask indicating (>0) pixels included in local neighborhoods. If None,
        the entire image is considered.
    shift_x, shift_y : int, optional
        Offsets applied to the footprint center; shifts are bounded so the center
        remains inside the footprint.
    s0, s1 : int, optional
        Radiometric interval bounds: pixels with values in [g - s0, g + s1] are
        considered for the local mean, where g is the center pixel value.
    
    Returns
    -------
    out : ndarray of shape (M, N), same dtype as input `image`
        Filtered image containing the local bilateral means.
    
    See also
    --------
    skimage.restoration.denoise_bilateral
    """

    return _apply(
        bilateral_cy._mean,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        s0=s0,
        s1=s1,
    )


def pop_bilateral(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10
):
    """
    Compute the local population count of pixels within the spatial footprint and the radiometric interval around each center pixel.
    
    The count for each center pixel is the number of neighboring pixels that are inside the footprint, inside the mask (if provided), and have a graylevel in the interval [g - s0, g + s1], where g is the center pixel's grayvalue.
    
    Parameters
    ----------
    image : ndarray of shape (M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray of shape (m, n)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray of shape (M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray, optional
        Mask array that defines (>0) area of the image included in the local neighborhood. If None, the complete image is used.
    shift_x, shift_y : int, optional
        Offset added to the footprint center point. Shift is bounded to the footprint sizes (center must be inside the given footprint).
    s0, s1 : int, optional
        Define the [s0, s1] interval around the grayvalue of the center pixel to be considered when counting neighbors.
    
    Returns
    -------
    out : ndarray of shape (M, N), same dtype as input `image`
        Output image where each pixel equals the number of neighbors meeting the footprint, mask, and graylevel-interval criteria.
    """

    return _apply(
        bilateral_cy._pop,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        s0=s0,
        s1=s1,
    )


def sum_bilateral(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, s0=10, s1=10
):
    """Apply a flat kernel bilateral filter.

    This is an edge-preserving and noise reducing denoising filter. It averages
    pixels based on their spatial closeness and radiometric similarity.

    Spatial closeness is measured by considering only the local pixel
    neighborhood given by a footprint (structuring element).

    Radiometric similarity is defined by the graylevel interval [g-s0, g+s1]
    where g is the current pixel graylevel.

    Only pixels belonging to the footprint AND having a graylevel inside this
    interval are summed.

    Note that the sum may overflow depending on the data type of the input
    array.

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
    s0, s1 : int
        Define the [s0, s1] interval around the grayvalue of the center pixel
        to be considered for computing the value.

    Returns
    -------
    out : ndarray of shape (M, N), same dtype as input `image`
        Output image.

    See also
    --------
    skimage.restoration.denoise_bilateral

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filters.rank import sum_bilateral
    >>> img = data.camera().astype(np.uint16)
    >>> bilat_img = sum_bilateral(img, disk(10), s0=10, s1=10)

    """

    return _apply(
        bilateral_cy._sum,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        s0=s0,
        s1=s1,
    )