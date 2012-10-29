import warnings
from skimage import img_as_ubyte

from . import cmorph


__all__ = ['erosion', 'dilation', 'opening', 'closing', 'white_tophat',
           'black_tophat', 'greyscale_erode', 'greyscale_dilate',
           'greyscale_open', 'greyscale_close', 'greyscale_white_top_hat',
           'greyscale_black_top_hat']


def erosion(image, selem, out=None, shift_x=False, shift_y=False):
    """Return greyscale morphological erosion of an image.

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
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).

    Returns
    -------
    eroded : uint8 array
        The result of the morphological erosion.

    Examples
    --------
    >>> # Erosion shrinks bright regions
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> bright_square = np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> erosion(bright_square, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """

    if image is out:
        raise NotImplementedError("In-place erosion not supported!")
    image = img_as_ubyte(image)
    selem = img_as_ubyte(selem)
    return cmorph.erode(image, selem, out=out,
                        shift_x=shift_x, shift_y=shift_y)


def dilation(image, selem, out=None, shift_x=False, shift_y=False):
    """Return greyscale morphological dilation of an image.

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
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).

    Returns
    -------
    dilated : uint8 array
        The result of the morphological dilation.

    Examples
    --------
    >>> # Dilation enlarges bright regions
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> bright_pixel = np.array([[0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 0, 1, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> dilation(bright_pixel, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """

    if image is out:
        raise NotImplementedError("In-place dilation not supported!")
    image = img_as_ubyte(image)
    selem = img_as_ubyte(selem)
    return cmorph.dilate(image, selem, out=out,
                         shift_x=shift_x, shift_y=shift_y)


def opening(image, selem, out=None):
    """Return greyscale morphological opening of an image.

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
    opening : uint8 array
        The result of the morphological opening.

    Examples
    --------
    >>> # Open up gap between two bright regions (but also shrink regions)
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> bad_connection = np.array([[1, 0, 0, 0, 1],
    ...                            [1, 1, 0, 1, 1],
    ...                            [1, 1, 1, 1, 1],
    ...                            [1, 1, 0, 1, 1],
    ...                            [1, 0, 0, 0, 1]], dtype=np.uint8)
    >>> opening(bad_connection, square(3))
    array([[0, 0, 0, 0, 0],
           [1, 1, 0, 1, 1],
           [1, 1, 0, 1, 1],
           [1, 1, 0, 1, 1],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """

    h, w = selem.shape
    shift_x = True if (w % 2) == 0 else False
    shift_y = True if (h % 2) == 0 else False

    eroded = erosion(image, selem)
    out = dilation(eroded, selem, out=out, shift_x=shift_x, shift_y=shift_y)
    return out


def closing(image, selem, out=None):
    """Return greyscale morphological closing of an image.

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
    closing : uint8 array
        The result of the morphological closing.

    Examples
    --------
    >>> # Close a gap between two bright lines
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> broken_line = np.array([[0, 0, 0, 0, 0],
    ...                         [0, 0, 0, 0, 0],
    ...                         [1, 1, 0, 1, 1],
    ...                         [0, 0, 0, 0, 0],
    ...                         [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> closing(broken_line, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """

    h, w = selem.shape
    shift_x = True if (w % 2) == 0 else False
    shift_y = True if (h % 2) == 0 else False

    dilated = dilation(image, selem)
    out = erosion(dilated, selem, out=out, shift_x=shift_x, shift_y=shift_y)
    return out


def white_tophat(image, selem, out=None):
    """Return white top hat of an image.

    The white top hat of an image is defined as the image minus its
    morphological opening. This operation returns the bright spots of the image
    that are smaller than the structuring element.

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
    opening : uint8 array
        The result of the morphological white top hat.

    Examples
    --------
    >>> # Subtract grey background from bright peak
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> bright_on_grey = np.array([[2, 3, 3, 3, 2],
    ...                            [3, 4, 5, 4, 3],
    ...                            [3, 5, 9, 5, 3],
    ...                            [3, 4, 5, 4, 3],
    ...                            [2, 3, 3, 3, 2]], dtype=np.uint8)
    >>> white_tophat(bright_on_grey, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 5, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

   """
    if image is out:
        raise NotImplementedError("Cannot perform white top hat in place.")

    out = opening(image, selem, out=out)
    out = image - out
    return out


def black_tophat(image, selem, out=None):
    """Return black top hat of an image.

    The black top hat of an image is defined as its morphological closing minus
    the original image. This operation returns the dark spots of the image that
    are smaller than the structuring element. Note that dark spots in the
    original image are bright spots after the black top hat.

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
    opening : uint8 array
       The result of the black top filter.

    Examples
    --------
    >>> # Change dark peak to bright peak and subtract background
    >>> import numpy as np
    >>> from skimage.morphology import square
    >>> dark_on_grey = np.array([[7, 6, 6, 6, 7],
    ...                          [6, 5, 4, 5, 6],
    ...                          [6, 4, 0, 4, 6],
    ...                          [6, 5, 4, 5, 6],
    ...                          [7, 6, 6, 6, 7]], dtype=np.uint8)
    >>> black_tophat(dark_on_grey, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 5, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """

    if image is out:
        raise NotImplementedError("Cannot perform white top hat in place.")

    out = closing(image, selem, out=out)
    out = out - image
    return out


def greyscale_erode(*args, **kwargs):
    warnings.warn("`greyscale_erode` renamed `erosion`.")
    return erosion(*args, **kwargs)


def greyscale_dilate(*args, **kwargs):
    warnings.warn("`greyscale_dilate` renamed `dilation`.")
    return dilation(*args, **kwargs)


def greyscale_open(*args, **kwargs):
    warnings.warn("`greyscale_open` renamed `opening`.")
    return opening(*args, **kwargs)


def greyscale_close(*args, **kwargs):
    warnings.warn("`greyscale_close` renamed `closing`.")
    return closing(*args, **kwargs)


def greyscale_white_top_hat(*args, **kwargs):
    warnings.warn("`greyscale_white_top_hat` renamed `white_tophat`.")
    return white_tophat(*args, **kwargs)


def greyscale_black_top_hat(*args, **kwargs):
    warnings.warn("`greyscale_black_top_hat` renamed `black_tophat`.")
    return black_tophat(*args, **kwargs)
