"""
:author: Olivier Debeir, 2012
:license: modified BSD
"""

__docformat__ = 'restructuredtext en'

import warnings
from skimage import img_as_ubyte

__all__ = ['percentile_mean']


def percentile_mean(image, selem, out=None, shift_x=False, shift_y=False, p0=.0, p1=1.):
    """Return greyscale local mean of an image.

    Mean is computed on the given structuring element. Only pixel values contained inside the
    percentile interval [p0,p1] are taken into account.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).
        Shift is bounded to the structuring element sizes.

    Returns
    -------
    local mean : uint8 array or uint16 array depending on input image
        The result of the local mean.

    Examples
    --------
    to be updated
    >>> # Erosion shrinks bright regions
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
    pass

