"""
Binary morphological operations
"""
import numpy as np
from scipy import ndimage as ndi
from .misc import default_selem


# The default_selem decorator provides a diamond structuring element as default
# with the same dimension as the input image and size 3 along each axis.
@default_selem
def binary_erosion(image, selem=None, out=None, iterations=1, mask=None,
                   origin=0, border_value=0, brute_force=False):
    """Return fast binary morphological erosion of an image.

    This function returns the same result as greyscale erosion but performs
    faster for binary images. It is a wrapper to the
    scipy.ndimage.binary_erosion function.

    Morphological erosion sets a pixel at ``(i,j)`` to the minimum over all
    pixels in the neighborhood centered at ``(i,j)``. Erosion shrinks bright
    regions and enlarges dark regions.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    selem : ndarray, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use cross-shaped structuring element (connectivity=1).
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    iterations : {int, float}, optional
        The number of times erosion is repeated (defaults to 1). If <1,
        dilation is repeated until the result no longer changes.
    mask : array-like, optional
        If provided, erosion is only performed on elements of `image` with
        a True value at the corresponding `mask` element.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    border_value : int (cast to 0 or 1), optional
        Value at the border of the output array.

    Returns
    -------
    eroded : ndarray of bool or uint
        The result of the morphological erosion taking values in
        ``[False, True]``.

    """
    out = ndi.binary_erosion(image, structure=selem, output=out,
                             iterations=iterations, mask=mask, origin=origin,
                             border_value=border_value, brute_force=brute_force)
    return out


@default_selem
def binary_dilation(image, selem=None, out=None, iterations=1, mask=None,
                    border_value=0, origin=0, brute_force=False):
    """Return fast binary morphological dilation of an image.

    This function returns the same result as greyscale dilation but performs
    faster for binary images. It is a wrapper for the
    scipy.ndimage.binary_dilation function.

    Morphological dilation sets a pixel at ``(i,j)`` to the maximum over all
    pixels in the neighborhood centered at ``(i,j)``. Dilation enlarges bright
    regions and shrinks dark regions.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    selem : ndarray, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use cross-shaped structuring element (connectivity=1).
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None, is
        passed, a new array will be allocated.
    iterations : {int, float}, optional
        The number of times dilation is repeated (defaults to 1). If < 1,
        dilation is repeated until the result no longer changes.
    mask : array-like, optional
        If provided, dilation is only performed on elements of `image` with
        a True value at the corresponding `mask` element.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    border_value : int (cast to 0 or 1), optional
        Value at the border of the output array.

    Returns
    -------
    dilated : ndarray of bool or uint
        The result of the morphological dilation with values in
        ``[False, True]``.

    """
    out = ndi.binary_dilation(image, structure=selem, output=out,
                              iterations=iterations, mask=mask, origin=origin,
                              border_value=border_value, brute_force=False)
    return out


@default_selem
def binary_opening(image, selem=None, out=None, iterations=1, origin=0):
    """Return fast binary morphological opening of an image.

    This function returns the same result as greyscale opening but performs
    faster for binary images. It is a wrapper for the
    scipy.ndimage.binary_opening function.

    The morphological opening on an image is defined as an erosion followed by
    a dilation. Opening can remove small bright spots (i.e. "salt") and connect
    small dark cracks. This tends to "open" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    selem : ndarray, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use cross-shaped structuring element (connectivity=1).
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None
        is passed, a new array will be allocated.
    iterations : {int, float}, optional
        The number of times binary closing (dilation followed by erosion = 1)
        is repeated. Defaults to 1. If < 1, closing is repeated until the
        output no longer changes.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.

    Returns
    -------
    opening : ndarray of bool
        The result of the morphological opening.

    """
    out = ndi.binary_opening(image, structure=selem, iterations=iterations,
                             output=out, origin=origin)
    return out


@default_selem
def binary_closing(image, selem=None, out=None, iterations=1, origin=0):
    """Return fast binary morphological closing of an image.

    This function acts as a wrapper for scipy.ndimage.binary_closing and
    returns the same result as greyscale closing but performs faster for
    binary images.

    The morphological closing on an image is defined as a dilation followed by
    an erosion. Closing can remove small dark spots (i.e. "pepper") and connect
    small bright cracks. This tends to "close" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    selem : ndarray, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use cross-shaped structuring element (connectivity=1).
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None,
        is passed, a new array will be allocated.
    iterations : {int, float}, optional
        The number of times binary closing (dilation followed by erosion = 1)
        is repeated. Defaults to 1. If < 1, closing is repeated until the
        output no longer changes.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.

    Returns
    -------
    closing : ndarray of bool
        The result of the morphological closing.

    """
    out = ndi.binary_closing(image, structure=selem, iterations=iterations,
                             output=out, origin=origin)
    return out
