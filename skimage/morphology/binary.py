import warnings
import numpy as np
from scipy import ndimage


def binary_erosion(image, selem, out=None):
    """Return fast binary morphological erosion of an image.

    This function returns the same result as greyscale erosion but performs
    faster for binary images.

    Morphological erosion sets a pixel at ``(i,j)`` to the minimum over all
    pixels in the neighborhood centered at ``(i,j)``. Erosion shrinks bright
    regions and enlarges dark regions.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray of bool
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.

    Returns
    -------
    eroded : ndarray of bool or uint
        The result of the morphological erosion with values in ``[0, 1]``.

    """
    selem = (selem != 0)
    selem_sum = np.sum(selem)

    if selem_sum <= 255:
        conv = np.empty_like(image, dtype=np.uint8)
    else:
        conv = np.empty_like(image, dtype=np.uint)

    binary = (image > 0).view(np.uint8)
    ndimage.convolve(binary, selem, mode='constant', cval=1, output=conv)

    if out is None:
        out = conv
    return np.equal(conv, selem_sum, out=out)


def binary_dilation(image, selem, out=None):
    """Return fast binary morphological dilation of an image.

    This function returns the same result as greyscale dilation but performs
    faster for binary images.

    Morphological dilation sets a pixel at ``(i,j)`` to the maximum over all
    pixels in the neighborhood centered at ``(i,j)``. Dilation enlarges bright
    regions and shrinks dark regions.

    Parameters
    ----------

    image : ndarray
        Binary input image.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray of bool
        The array to store the result of the morphology. If None, is
        passed, a new array will be allocated.

    Returns
    -------
    dilated : ndarray of bool or uint
        The result of the morphological dilation with values in ``[0, 1]``.

    """
    selem = (selem != 0)

    if np.sum(selem) <= 255:
        conv = np.empty_like(image, dtype=np.uint8)
    else:
        conv = np.empty_like(image, dtype=np.uint)

    binary = (image > 0).view(np.uint8)
    ndimage.convolve(binary, selem, mode='constant', cval=0, output=conv)

    if out is None:
        out = conv
    return np.not_equal(conv, 0, out=out)


def binary_opening(image, selem, out=None):
    """Return fast binary morphological opening of an image.

    This function returns the same result as greyscale opening but performs
    faster for binary images.

    The morphological opening on an image is defined as an erosion followed by
    a dilation. Opening can remove small bright spots (i.e. "salt") and connect
    small dark cracks. This tends to "open" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray of bool
        The array to store the result of the morphology. If None
        is passed, a new array will be allocated.

    Returns
    -------
    opening : ndarray of bool
        The result of the morphological opening.

    """
    eroded = binary_erosion(image, selem)
    out = binary_dilation(eroded, selem, out=out)
    return out


def binary_closing(image, selem, out=None):
    """Return fast binary morphological closing of an image.

    This function returns the same result as greyscale closing but performs
    faster for binary images.

    The morphological closing on an image is defined as a dilation followed by
    an erosion. Closing can remove small dark spots (i.e. "pepper") and connect
    small bright cracks. This tends to "close" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray of bool
        The array to store the result of the morphology. If None,
        is passed, a new array will be allocated.

    Returns
    -------
    closing : ndarray of bool
        The result of the morphological closing.

    """

    dilated = binary_dilation(image, selem)
    out = binary_erosion(dilated, selem, out=out)
    return out
