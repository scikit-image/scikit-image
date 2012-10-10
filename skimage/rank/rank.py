"""rank.py - rankfilter for local (custom kernel) maximum, minimum, median, mean, auto-level, egalize, etc

The local histogram is computed using a sliding window similar to the method described in

Reference: Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional median filtering algorithm",
IEEE Transactions on Acoustics, Speech and Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

:author: Olivier Debeir, 2012
:license: modified BSD
"""

__docformat__ = 'restructuredtext en'

import warnings
from skimage import img_as_ubyte
import numpy as np

from generic import find_bitdepth
import _crank16,_crank8

__all__ = ['autolevel','bottomhat','egalise','gradient','maximum','mean'
    ,'meansubstraction','median','minimum','modal','morph_contr_enh','pop','threshold', 'tophat']

def autolevel(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local autolevel of an image.

    Autolevel is computed on the given structuring element.

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

    Returns
    -------
    local autolevel : uint8 array or uint16 array depending on input image
        The result of the local autolevel.

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
    >>> autolevel(ima8, square(3))
    array([[  0,   0,   0,   0,   0],
           [  0, 255, 255, 255,   0],
           [  0, 255,   0, 255,   0],
           [  0, 255, 255, 255,   0],
           [  0,   0,   0,   0,   0]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> autolevel(ima16, square(3))
    array([[   0,    0,    0,    0,    0],
           [   0, 4096, 4096, 4096,    0],
           [   0, 4096,    0, 4096,    0],
           [   0, 4096, 4096, 4096,    0],
           [   0,    0,    0,    0,    0]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8.autolevel(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.autolevel(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def bottomhat(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local bottomhat of an image.

    Bottomhat is computed on the given structuring element.

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

    Returns
    -------
    local bottomhat : uint8 array or uint16 array depending on input image
        The result of the local bottomhat.

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
    >>> bottomhat(ima8, square(3))
    array([[  0,   0,   0,   0,   0],
           [  0, 255, 255, 255,   0],
           [  0, 255,   0, 255,   0],
           [  0, 255, 255, 255,   0],
           [  0,   0,   0,   0,   0]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> bottomhat(ima16, square(3))
    array([[   0,    0,    0,    0,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0, 4095,    0, 4095,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0,    0,    0,    0,    0]], dtype=uint16)
    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8.bottomhat(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.bottomhat(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def egalise(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local egalise of an image.

    egalise is computed on the given structuring element.

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

    Returns
    -------
    local egalise : uint8 array or uint16 array depending on input image
        The result of the local egalise.

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
    >>> egalise(ima8, square(3))
    array([[191, 170, 127, 170, 191],
           [170, 255, 255, 255, 170],
           [127, 255, 255, 255, 127],
           [170, 255, 255, 255, 170],
           [191, 170, 127, 170, 191]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> egalise(ima16, square(3))
    array([[3072, 2730, 2048, 2730, 3072],
           [2730, 4096, 4096, 4096, 2730],
           [2048, 4096, 4096, 4096, 2048],
           [2730, 4096, 4096, 4096, 2730],
           [3072, 2730, 2048, 2730, 3072]], dtype=uint16)
    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8.egalise(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.egalise(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def gradient(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local gradient of an image.

    gradient is computed on the given structuring element.

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

    Returns
    -------
    local gradient : uint8 array or uint16 array depending on input image
        The result of the local gradient.

    Examples
    --------
    to be updated
    >>> # Local gradient
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> gradient(ima8, square(3))
    array([[255, 255, 255, 255, 255],
           [255, 255, 255, 255, 255],
           [255, 255,   0, 255, 255],
           [255, 255, 255, 255, 255],
           [255, 255, 255, 255, 255]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> gradient(ima16, square(3))
    array([[4095, 4095, 4095, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095],
           [4095, 4095,    0, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8.gradient(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.gradient(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")


def maximum(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local maximum of an image.

    maximum is computed on the given structuring element.

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

    Returns
    -------
    local maximum : uint8 array or uint16 array depending on input image
        The result of the local maximum.

    Examples
    --------
    to be updated
    >>> # Local maximum
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 0, 0, 0, 0],
    ...                           [0, 0, 1, 0, 0],
    ...                           [0, 0, 0, 0, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> maximum(ima8, square(3))
    array([[  0,   0,   0,   0,   0],
           [  0, 255, 255, 255,   0],
           [  0, 255, 255, 255,   0],
           [  0, 255, 255, 255,   0],
           [  0,   0,   0,   0,   0]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 0, 0, 0, 0],
    ...                           [0, 0, 1, 0, 0],
    ...                           [0, 0, 0, 0, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> maximum(ima16, square(3))
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
        return _crank8.maximum(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.maximum(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def mean(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local mean of an image.

    Mean is computed on the given structuring element.

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

    Returns
    -------
    local mean : uint8 array or uint16 array depending on input image
        The result of the local mean.

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
    >>> mean(ima8, square(3))
    array([[ 63,  85, 127,  85,  63],
           [ 85, 113, 170, 113,  85],
           [127, 170, 255, 170, 127],
           [ 85, 113, 170, 113,  85],
           [ 63,  85, 127,  85,  63]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> mean(ima16, square(3))
    array([[1023, 1365, 2047, 1365, 1023],
           [1365, 1820, 2730, 1820, 1365],
           [2047, 2730, 4095, 2730, 2047],
           [1365, 1820, 2730, 1820, 1365],
           [1023, 1365, 2047, 1365, 1023]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8.mean(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.mean(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def meansubstraction(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local meansubstraction of an image.

    meansubstraction is computed on the given structuring element.

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

    Returns
    -------
    local meansubstraction : uint8 array or uint16 array depending on input image
        The result of the local meansubstraction.

    Examples
    --------
    to be updated
    >>> # Local meansubstraction
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> meansubstraction(ima8, square(3))
    array([[ 95,  84,  63,  84,  95],
           [ 84, 197, 169, 197,  84],
           [ 63, 169, 127, 169,  63],
           [ 84, 197, 169, 197,  84],
           [ 95,  84,  63,  84,  95]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> meansubstraction(ima16, square(3))
    array([[1536, 1365, 1024, 1365, 1536],
           [1365, 3185, 2730, 3185, 1365],
           [1024, 2730, 2048, 2730, 1024],
           [1365, 3185, 2730, 3185, 1365],
           [1536, 1365, 1024, 1365, 1536]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8.meansubstraction(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.meansubstraction(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def median(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local median of an image.

    median is computed on the given structuring element.

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

    Returns
    -------
    local median : uint8 array or uint16 array depending on input image
        The result of the local median.

    Examples
    --------
    to be updated
    >>> # Local median
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 0, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> median(ima8, square(3))
    array([[  0,   0, 255,   0,   0],
           [  0,   0, 255,   0,   0],
           [255, 255, 255, 255, 255],
           [  0,   0, 255,   0,   0],
           [  0,   0, 255,   0,   0]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 0, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> median(ima16, square(3))
    array([[   0,    0, 4095,    0,    0],
           [   0,    0, 4095,    0,    0],
           [4095, 4095, 4095, 4095, 4095],
           [   0,    0, 4095,    0,    0],
           [   0,    0, 4095,    0,    0]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8.median(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.median(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def minimum(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local minimum of an image.

    minimum is computed on the given structuring element.

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

    Returns
    -------
    local minimum : uint8 array or uint16 array depending on input image
        The result of the local minimum.

    Examples
    --------
    to be updated
    >>> # Local minimum
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> minimum(ima8, square(3))
    array([[  0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0],
           [  0,   0, 255,   0,   0],
           [  0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0]], dtype=uint8)


    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> minimum(ima16, square(3))
    array([[   0,    0,    0,    0,    0],
           [   0,    0,    0,    0,    0],
           [   0,    0, 4095,    0,    0],
           [   0,    0,    0,    0,    0],
           [   0,    0,    0,    0,    0]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8.minimum(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.minimum(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def modal(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local modal of an image.

    modal is computed on the given structuring element.

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

    Returns
    -------
    local modal : uint8 array or uint16 array depending on input image
        The result of the local modal.

    Examples
    --------
    to be updated
    >>> # Local modal
    >>> from skimage.morphology import square
    >>> ima8 = np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 5, 6, 0],
    ...                           [0, 1, 5, 5, 0],
    ...                           [0, 0, 0, 5, 0]], dtype=np.uint8)
    >>> modal(ima8, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 1, 0, 0],
           [0, 0, 5, 0, 0],
           [0, 0, 5, 0, 0]], dtype=uint8)


    >>> ima16 = 100*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 5, 6, 0],
    ...                           [0, 1, 5, 5, 0],
    ...                           [0, 0, 0, 5, 0]], dtype=np.uint16)
    >>> modal(ima16, square(3))
    array([[  0,   0,   0,   0,   0],
           [  0,   0, 100,   0,   0],
           [  0, 100, 100,   0,   0],
           [  0,   0, 500,   0,   0],
           [  0,   0, 500,   0,   0]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8.modal(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.modal(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def morph_contr_enh(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local morph_contr_enh of an image.

    morph_contr_enh is computed on the given structuring element.

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

    Returns
    -------
    local morph_contr_enh : uint8 array or uint16 array depending on input image
        The result of the local morph_contr_enh.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> ima8 = np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> morph_contr_enh(ima8, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> morph_contr_enh(ima16, square(3))
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
        return _crank8.morph_contr_enh(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.morph_contr_enh(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def pop(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local pop of an image.

    pop is computed on the given structuring element.

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

    Returns
    -------
    local pop : uint8 array or uint16 array depending on input image
        The result of the local pop.

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
    >>> pop(ima8, square(3))
    array([[4, 6, 6, 6, 4],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [4, 6, 6, 6, 4]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> pop(ima16, square(3))
    array([[4, 6, 6, 6, 4],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [4, 6, 6, 6, 4]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8.pop(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.pop(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def threshold(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local threshold of an image.

    threshold is computed on the given structuring element.

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

    Returns
    -------
    local threshold : uint8 array or uint16 array depending on input image
        The result of the local threshold.

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
    >>> threshold(ima8, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> threshold(ima16, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint16)


    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8.threshold(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.threshold(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def tophat(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local tophat of an image.

    tophat is computed on the given structuring element.

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

    Returns
    -------
    local tophat : uint8 array or uint16 array depending on input image
        The result of the local tophat.

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
    >>> tophat(ima8, square(3))
    array([[255, 255, 255, 255, 255],
           [255,   0,   0,   0, 255],
           [255,   0,   0,   0, 255],
           [255,   0,   0,   0, 255],
           [255, 255, 255, 255, 255]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> tophat(ima16, square(3))
    array([[4095, 4095, 4095, 4095, 4095],
           [4095,    0,    0,    0, 4095],
           [4095,    0,    0,    0, 4095],
           [4095,    0,    0,    0, 4095],
           [4095, 4095, 4095, 4095, 4095]], dtype=uint16)
    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8.tophat(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16.tophat(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")