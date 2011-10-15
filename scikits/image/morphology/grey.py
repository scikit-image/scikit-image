"""
:author: Damian Eads, 2009
:license: modified BSD
"""

__docformat__ = 'restructuredtext en'

import numpy as np

eps = np.finfo(float).eps

def greyscale_erode(image, selem, out=None, flip=False):
    """
    Performs a greyscale morphological erosion on an image given a flat
    structuring element. The eroded pixel at (i,j) is the minimum over all
    pixels in the neighborhood centered at (i,j).

    Parameters
    ----------
    image : ndarray
       The image as an ndarray.

    selem : ndarray
       The neighborhood expressed as a 2-D array of 1's and 0's.

    out : ndarray
       The array to store the result of the morphology. If None is
       passed, a new array will be allocated.

    flip : bool
        Flip structuring element about center point. This only affects
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
        out = cmorph.erode(image, selem, out=out, flip=flip)
        return out;
    except ImportError:
        raise ImportError("cmorph extension not available.")

def greyscale_dilate(image, selem, out=None, flip=False):
    """
    Performs a greyscale morphological dilation on an image given a flat
    structuring element. The dilated pixel at (i,j) is the maximum over all
    pixels in the neighborhood centered at (i,j).

    Parameters
    ----------

    image : ndarray
       The image as an ndarray.

    selem : ndarray
       The neighborhood expressed as a 2-D array of 1's and 0's.

    out : ndarray
       The array to store the result of the morphology. If None, is
       passed, a new array will be allocated.

    flip : bool
        Flip structuring element about center point. This only affects
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
        out = cmorph.dilate(image, selem, out=out, flip=flip)
        return out;
    except ImportError:
        raise ImportError("cmorph extension not available.")

def greyscale_open(image, selem, out=None):
    """
    Performs a greyscale morphological opening on an image given a flat
    structuring element defined as a erosion followed by a dilation.


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
    eroded = greyscale_erode(image, selem)
    out = greyscale_dilate(eroded, selem, out=out)
    return out

def greyscale_close(image, selem, out=None):
    """
    Performs a greyscale morphological closing on an image given a flat
    structuring element defined as a dilation followed by an erosion.

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
    dilated = greyscale_dilate(image, selem)
    out = greyscale_erode(dilated, selem, out=out)
    return out

def greyscale_white_top_hat(image, selem, out=None):
    """
    Applies a white top hat on an image given a flat structuring element.

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
    """
    Applies a black top hat on an image given a flat structuring element.

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
