"""
:author: Olivier Debeir, 2012
:license: modified BSD
"""

__docformat__ = 'restructuredtext en'

import warnings
from skimage import img_as_ubyte

import numpy as np

from generic import find_bitdepth
import _crank16_bilateral


__all__ = ['bilateral_mean']


def bilateral_mean(image, selem, out=None, mask=None, shift_x=False, shift_y=False, s0=10, s1=10):
    """Return greyscale local bilateral_mean of an image.

    bilateral mean is computed on the given structuring element. Only levels between [g-s0,g+s1] ,are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, as the algorithm uses max. 12bit histogram,
        an exception will be raised if image has a value > 4095
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local neighborhood.
        If None, the complete image is used (default).
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).
        Shift is bounded to the structuring element sizes.
    s0, s1 : int
        define the [s0,s1] interval to be considered for computing the value.

    Returns
    -------
    local bilateral mean : uint16 array (uint8 image are casted to uint16)
        The result of the local bilateral mean.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> bilateral_mean(ima8, square(3), s0=10,s1=10)
    array([[  0,   0,   0,   0,   0],
           [  0, 255, 255, 255,   0],
           [  0, 255, 255, 255,   0],
           [  0, 255, 255, 255,   0],
           [  0,   0,   0,   0,   0]], dtype=uint16)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> bilateral_mean(ima16, square(3), s0=10,s1=10)
    array([[   0,    0,    0,    0,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0,    0,    0,    0,    0]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        image = image.astype(np.uint16)
    elif image.dtype == np.uint16:
        pass
    else:
        raise TypeError("only uint8 and uint16 image supported!")
    bitdepth = find_bitdepth(image)
    if bitdepth>11:
        raise ValueError("only uint16 <4096 image (12bit) supported!")
    return _crank16_bilateral.mean(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out,s0=s0,s1=s1)


def bilateral_pop(image, selem, out=None, mask=None, shift_x=False, shift_y=False, s0=10, s1=10):
    """Return greyscale local bilateral_pop of an image.

    bilateral pop is computed on the given structuring element. Only levels between [g-s0,g+s1] ,are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, as the algorithm uses max. 12bit histogram,
        an exception will be raised if image has a value > 4095
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local neighborhood.
        If None, the complete image is used (default).
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).
        Shift is bounded to the structuring element sizes.
    s0, s1 : int
        define the [s0,s1] interval to be considered for computing the value.

    Returns
    -------
    local bilateral pop : uint16 array (uint8 image are casted to uint16)
        The result of the local bilateral pop.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> bilateral_pop(ima8, square(3), s0=10,s1=10)
    array([[3, 4, 3, 4, 3],
           [4, 4, 6, 4, 4],
           [3, 6, 9, 6, 3],
           [4, 4, 6, 4, 4],
           [3, 4, 3, 4, 3]], dtype=uint16)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> bilateral_pop(ima16, square(3), s0=10,s1=10)
    array([[3, 4, 3, 4, 3],
           [4, 4, 6, 4, 4],
           [3, 6, 9, 6, 3],
           [4, 4, 6, 4, 4],
           [3, 4, 3, 4, 3]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        image = image.astype(np.uint16)
    elif image.dtype == np.uint16:
        pass
    else:
        raise TypeError("only uint8 and uint16 image supported!")
    bitdepth = find_bitdepth(image)
    if bitdepth>11:
        raise ValueError("only uint16 <4096 image (12bit) supported!")
    return _crank16_bilateral.pop(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out,s0=s0,s1=s1)


