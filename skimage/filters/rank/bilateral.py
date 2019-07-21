"""Approximate bilateral rank filter for local (custom kernel) mean.

The local histogram is computed using a sliding window similar to the method
described in [1]_.

The pixel neighborhood is defined by:

* the given structuring element
* an interval [g-s0, g+s1] in greylevel around g the processed pixel greylevel

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

import numpy as np

from ..._shared.utils import check_nD
from . import bilateral_cy
from .generic import _handle_input


__all__ = ['mean_bilateral', 'pop_bilateral', 'sum_bilateral']


def _apply(func, image, selem, out, mask, shift_x, shift_y, s0, s1,
           out_dtype=None):

    check_nD(image, 2)
    image, selem, out, mask, n_bins = _handle_input(image, selem, out, mask,
                                                    out_dtype)

    func(image, selem, shift_x=shift_x, shift_y=shift_y, mask=mask,
         out=out, n_bins=n_bins, s0=s0, s1=s1)

    return out.reshape(out.shape[:2])


def mean_bilateral(image, selem, out=None, mask=None, shift_x=False,
                   shift_y=False, s0=10, s1=10):
    """Apply a flat kernel bilateral filter.

    This is an edge-preserving and noise reducing denoising filter. It averages
    pixels based on their spatial closeness and radiometric similarity.

    Spatial closeness is measured by considering only the local pixel
    neighborhood given by a structuring element.

    Radiometric similarity is defined by the greylevel interval [g-s0, g+s1]
    where g is the current pixel greylevel.

    Only pixels belonging to the structuring element and having a greylevel
    inside this interval are averaged.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    s0, s1 : int
        Define the [s0, s1] interval around the greyvalue of the center pixel
        to be considered for computing the value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    See also
    --------
    denoise_bilateral

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filters.rank import mean_bilateral
    >>> img = data.camera().astype(np.uint16)
    >>> bilat_img = mean_bilateral(img, disk(20), s0=10,s1=10)

    """

    return _apply(bilateral_cy._mean, image, selem, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y, s0=s0, s1=s1)


def pop_bilateral(image, selem, out=None, mask=None, shift_x=False,
                  shift_y=False, s0=10, s1=10):
    """Return the local number (population) of pixels.


    The number of pixels is defined as the number of pixels which are included
    in the structuring element and the mask. Additionally pixels must have a
    greylevel inside the interval [g-s0, g+s1] where g is the greyvalue of the
    center pixel.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    s0, s1 : int
        Define the [s0, s1] interval around the greyvalue of the center pixel
        to be considered for computing the value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage.morphology import square
    >>> import skimage.filters.rank as rank
    >>> img = 255 * np.array([[0, 0, 0, 0, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> rank.pop_bilateral(img, square(3), s0=10, s1=10)
    array([[3, 4, 3, 4, 3],
           [4, 4, 6, 4, 4],
           [3, 6, 9, 6, 3],
           [4, 4, 6, 4, 4],
           [3, 4, 3, 4, 3]], dtype=uint16)

    """

    return _apply(bilateral_cy._pop, image, selem, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y, s0=s0, s1=s1)


def sum_bilateral(image, selem, out=None, mask=None, shift_x=False,
                  shift_y=False, s0=10, s1=10):
    """Apply a flat kernel bilateral filter.

    This is an edge-preserving and noise reducing denoising filter. It averages
    pixels based on their spatial closeness and radiometric similarity.

    Spatial closeness is measured by considering only the local pixel
    neighborhood given by a structuring element (selem).

    Radiometric similarity is defined by the greylevel interval [g-s0, g+s1]
    where g is the current pixel greylevel.

    Only pixels belonging to the structuring element AND having a greylevel
    inside this interval are summed.

    Note that the sum may overflow depending on the data type of the input
    array.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    s0, s1 : int
        Define the [s0, s1] interval around the greyvalue of the center pixel
        to be considered for computing the value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    See also
    --------
    denoise_bilateral

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filters.rank import sum_bilateral
    >>> img = data.camera().astype(np.uint16)
    >>> bilat_img = sum_bilateral(img, disk(10), s0=10, s1=10)

    """

    return _apply(bilateral_cy._sum, image, selem, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y, s0=s0, s1=s1)
