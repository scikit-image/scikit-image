"""
:author: Damian Eads, 2009
:license: modified BSD
"""

__docformat__ = 'restructuredtext en'

import numpy as np

eps = np.finfo(float).eps

def greyscale_erode(image, selem, out=None, shift_x=False, shift_y=False):
    """Return greyscale morphological erosion of an image.

    Morphological erosion sets a pixel at (i,j) to the minimum over all pixels
    in the neighborhood centered at (i,j). Erosion shrinks bright regions and
    enlarges dark regions.

    Parameters
    ----------
    image : ndarray
       The image as an ndarray.

    selem : ndarray
       The neighborhood expressed as a 2-D array of 1's and 0's.

    out : ndarray
       The array to store the result of the morphology. If None is
       passed, a new array will be allocated.

    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).

    Returns
    -------
    eroded : ndarray
       The result of the morphological erosion.
    """
    if image is out:
        raise NotImplementedError("In-place erosion not supported!")
    try:
        import scikits.image.morphology.cmorph as cmorph
        out = cmorph.erode(image, selem, out=out,
                           shift_x=shift_x, shift_y=shift_y)
        return out;
    except ImportError:
        raise ImportError("cmorph extension not available.")

def greyscale_dilate(image, selem, out=None, shift_x=False, shift_y=False):
    """Return greyscale morphological dilation of an image.

    Morphological dilation sets a pixel at (i,j) to the maximum over all pixels
    in the neighborhood centered at (i,j). Dilation enlarges bright regions
    and shrinks dark regions.

    Parameters
    ----------

    image : ndarray
       The image as an ndarray.

    selem : ndarray
       The neighborhood expressed as a 2-D array of 1's and 0's.

    out : ndarray
       The array to store the result of the morphology. If None, is
       passed, a new array will be allocated.

    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).

    Returns
    -------
    dilated : ndarray
       The result of the morphological dilation.
    """
    if image is out:
        raise NotImplementedError("In-place dilation not supported!")
    try:
        from . import cmorph
        out = cmorph.dilate(image, selem, out=out,
                            shift_x=shift_x, shift_y=shift_y)
        return out;
    except ImportError:
        raise ImportError("cmorph extension not available.")

def greyscale_open(image, selem, out=None):
    """Return greyscale morphological opening of an image.

    The morphological opening on an image is defined as an erosion followed by
    a dilation. Opening can remove small bright spots (i.e. "salt") and connect
    small dark cracks. This tends to "open" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
       The image as an ndarray.

    selem : ndarray
       The neighborhood expressed as a 2-D array of 1's and 0's.

    out : ndarray
       The array to store the result of the morphology. If None
       is passed, a new array will be allocated.

    Returns
    -------
    opening : ndarray
       The result of the morphological opening.
    """
    h, w = selem.shape
    shift_x = True if (w % 2) == 0 else False
    shift_y = True if (h % 2) == 0 else False

    eroded = greyscale_erode(image, selem)
    out = greyscale_dilate(eroded, selem, out=out,
                           shift_x=shift_x, shift_y=shift_y)
    return out

def greyscale_close(image, selem, out=None):
    """Return greyscale morphological closing of an image.

    The morphological closing on an image is defined as a dilation followed by
    an erosion. Closing can remove small dark spots (i.e. "pepper") and connect
    small bright cracks. This tends to "close" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
       The image as an ndarray.

    selem : ndarray
       The neighborhood expressed as a 2-D array of 1's and 0's.

    out : ndarray
       The array to store the result of the morphology. If None,
       is passed, a new array will be allocated.

    Returns
    -------
    opening : ndarray
       The result of the morphological opening.
    """
    h, w = selem.shape
    shift_x = True if (w % 2) == 0 else False
    shift_y = True if (h % 2) == 0 else False

    dilated = greyscale_dilate(image, selem)
    out = greyscale_erode(dilated, selem, out=out,
                          shift_x=shift_x, shift_y=shift_y)
    return out

def greyscale_white_top_hat(image, selem, out=None):
    """Return white top hat of an image.

    The white top hat of an image is defined as the image minus its
    morphological opening. This operation returns the bright spots of the image
    that are smaller than the structuring element.

    Parameters
    ----------
    image : ndarray
       The image as an ndarray.

    selem : ndarray
       The neighborhood expressed as a 2-D array of 1's and 0's.

    out : ndarray
       The array to store the result of the morphology. If None
       is passed, a new array will be allocated.

    Returns
    -------
    opening : ndarray
       The result of the morphological white top hat.
    """
    if image is out:
        raise NotImplementedError("Cannot perform white top hat in place.")

    out = greyscale_open(image, selem, out=out)
    out = image - out
    return out

def greyscale_black_top_hat(image, selem, out=None):
    """Return black top hat of an image.

    The black top hat of an image is defined as its morphological closing minus
    the original image. This operation returns the dark spots of the image that
    are smaller than the structuring element. Note that dark spots in the
    original image are bright spots after the black top hat.

    Parameters
    ----------
    image : ndarray
       The image as an ndarray.

    selem : ndarray
       The neighborhood expressed as a 2-D array of 1's and 0's.

    out : ndarray
       The array to store the result of the morphology. If None
       is passed, a new array will be allocated.

    Returns
    -------
    opening : ndarray
       The result of the black top filter.
    """
    if image is out:
        raise NotImplementedError("Cannot perform white top hat in place.")
    out = greyscale_close(image, selem, out=out)
    out = out - image
    return out
