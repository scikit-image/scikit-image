"""
Binary morphological operations
"""
import numpy as np
from scipy import ndimage as ndi


def isotropic_erosion(image, radius, out=None):
    """Return binary morphological erosion of an image.

    This function returns the same result as binary erosion but performs
    faster for large circular structuring elements.

    Morphological erosion sets a pixel at ``(i,j)`` to the minimum over all
    pixels in the neighborhood centered at ``(i,j)``. Erosion shrinks bright
    regions and enlarges dark regions.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius by which regions should be eroded.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.

    Returns
    -------
    eroded : ndarray of bool
        The result of the morphological erosion taking values in
        ``[False, True]``.
    """

    dist = ndi.distance_transform_edt(image)
    return np.greater(dist, radius, out=out)


def isotropic_dilation(image, radius, out=None):
    """Return binary morphological dilation of an image.

    This function returns the same result as binary dilation but performs
    faster for large circular structuring elements.

    Morphological dilation sets a pixel at ``(i,j)`` to the maximum over all
    pixels in the neighborhood centered at ``(i,j)``. Dilation enlarges bright
    regions and shrinks dark regions.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius by which regions should be dilated.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.

    Returns
    -------
    dilated : ndarray of bool
        The result of the morphological dilation with values in
        ``[False, True]``.
    """

    dist = ndi.distance_transform_edt(image == 0)
    return np.less(dist, radius, out=out)


def isotropic_opening(image, radius, out=None):
    """Return binary morphological opening of an image.

    This function returns the same result as binary opening but performs
    faster for large circular structuring elements.

    The morphological opening on an image is defined as an erosion followed by
    a dilation. Opening can remove small bright spots (i.e. "salt") and connect
    small dark cracks. This tends to "open" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius with which the regions should be opened.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None
        is passed, a new array will be allocated.

    Returns
    -------
    opening : ndarray of bool
        The result of the morphological opening.
    """

    eroded = isotropic_erosion(image, radius, out=out)
    return isotropic_dilation(eroded, radius, out=out)


def isotropic_closing(image, radius, out=None):
    """Return binary morphological closing of an image.

    This function returns the same result as binary closing but performs
    faster for large circular structuring elements.

    The morphological closing on an image is defined as a dilation followed by
    an erosion. Closing can remove small dark spots (i.e. "pepper") and connect
    small bright cracks. This tends to "close" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius with which the regions should be closed.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None,
        is passed, a new array will be allocated.

    Returns
    -------
    closing : ndarray of bool
        The result of the morphological closing.
    """

    dilated = isotropic_dilation(image, radius, out=out)
    return isotropic_erosion(dilated, radius, out=out)
