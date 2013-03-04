import numpy as np
from scipy import ndimage


def binary_erosion(image, selem, out=None):
    """Return fast binary morphological erosion of an image.

    This function returns the same result as greyscale erosion but performs
    faster for binary images.

    Morphological erosion sets a pixel at (i,j) to the minimum over all pixels
    in the neighborhood centered at (i,j). Erosion shrinks bright regions and
    enlarges dark regions.

    Parameters
    ----------
    image : ndarray
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.

    Returns
    -------
    eroded : bool array
        The result of the morphological erosion.

    """

    conv = ndimage.convolve(image > 0, selem, output=out,
                            mode='constant', cval=1)
    if conv is not None:
        out = conv
    return np.equal(out, np.sum(selem), out=out)


def binary_dilation(image, selem, out=None):
    """Return fast binary morphological dilation of an image.

    This function returns the same result as greyscale dilation but performs
    faster for binary images.

    Morphological dilation sets a pixel at (i,j) to the maximum over all pixels
    in the neighborhood centered at (i,j). Dilation enlarges bright regions
    and shrinks dark regions.

    Parameters
    ----------

    image : ndarray
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None, is
        passed, a new array will be allocated.

    Returns
    -------
    dilated : bool array
        The result of the morphological dilation.

    """

    conv = ndimage.convolve(image > 0, selem, output=out,
                            mode='constant', cval=0)
    if conv is not None:
        out = conv
    return np.not_equal(out, 0, out=out)


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
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None
        is passed, a new array will be allocated.

    Returns
    -------
    opening : bool array
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
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None,
        is passed, a new array will be allocated.

    Returns
    -------
    closing : bool array
        The result of the morphological closing.

    """

    dilated = binary_dilation(image, selem)
    out = binary_erosion(dilated, selem, out=out)
    return out
