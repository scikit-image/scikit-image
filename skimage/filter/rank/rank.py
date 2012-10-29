"""rank.py - rankfilter for local (custom kernel) maximum, minimum, median, mean, auto-level, equalization, etc

The local histogram is computed using a sliding window similar to the method described in

Reference: Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional median filtering algorithm",
IEEE Transactions on Acoustics, Speech and Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

input image can be 8 bit or 16 bit with a value < 4096 (i.e. 12 bit),
for 16 bit input images, the number of histogram bins is determined from the maximum value present in the image

result image is 8 or 16 bit with respect to the input image

"""

from skimage import img_as_ubyte
import numpy as np
from skimage.filter.rank import _crank8, _crank16

from skimage.filter.rank.generic import find_bitdepth

__all__ = ['autolevel', 'bottomhat', 'equalize', 'gradient', 'maximum', 'mean', 'meansubstraction', 'median', 'minimum',
           'modal', 'morph_contr_enh', 'pop', 'threshold', 'tophat']


def _apply(func8, func16, image, selem, out, mask, shift_x, shift_y):
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return func8(image, selem, shift_x=shift_x, shift_y=shift_y, mask=mask, out=out)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth > 11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return func16(image, selem, shift_x=shift_x, shift_y=shift_y, mask=mask, bitdepth=bitdepth + 1, out=out)
    else:
        raise TypeError("only uint8 and uint16 image supported!")


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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local autolevel : uint8 array or uint16 array depending on input image
        The result of the local autolevel.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.autolevel(ima8, square(3))
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
    >>> rank.autolevel(ima16, square(3))
    array([[   0,    0,    0,    0,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0, 4095,    0, 4095,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0,    0,    0,    0,    0]], dtype=uint16)

    """

    return _apply(
        _crank8.autolevel, _crank16.autolevel, image, selem, out=out, mask=mask, shift_x=shift_x,
        shift_y=shift_y)


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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local bottomhat : uint8 array or uint16 array depending on input image
        The result of the local bottomhat.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.bottomhat(ima8, square(3))
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
    >>> rank.bottomhat(ima16, square(3))
    array([[   0,    0,    0,    0,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0, 4095,    0, 4095,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0,    0,    0,    0,    0]], dtype=uint16)
    """

    return _apply(
        _crank8.bottomhat, _crank16.bottomhat, image, selem, out=out, mask=mask, shift_x=shift_x,
        shift_y=shift_y)


def equalize(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local equalize of an image.

    equalize is computed on the given structuring element.

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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local equalize : uint8 array or uint16 array depending on input image
        The result of the local equalize.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.equalize(ima8, square(3))
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
    >>> rank.equalize(ima16, square(3))
    array([[3071, 2730, 2047, 2730, 3071],
           [2730, 4095, 4095, 4095, 2730],
           [2047, 4095, 4095, 4095, 2047],
           [2730, 4095, 4095, 4095, 2730],
           [3071, 2730, 2047, 2730, 3071]], dtype=uint16)
    """

    return _apply(
        _crank8.equalize, _crank16.equalize, image, selem, out=out, mask=mask, shift_x=shift_x,
        shift_y=shift_y)


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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local gradient : uint8 array or uint16 array depending on input image
        The result of the local gradient.

    Examples
    --------
    to be updated
    >>> # Local gradient
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.gradient(ima8, square(3))
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
    >>> rank.gradient(ima16, square(3))
    array([[4095, 4095, 4095, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095],
           [4095, 4095,    0, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095]], dtype=uint16)

    """

    return _apply(
        _crank8.gradient, _crank16.gradient, image, selem, out=out, mask=mask, shift_x=shift_x,
        shift_y=shift_y)


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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local maximum : uint8 array or uint16 array depending on input image
        The result of the local maximum.

    Examples
    --------
    to be updated
    >>> # Local maximum
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 0, 0, 0, 0],
    ...                           [0, 0, 1, 0, 0],
    ...                           [0, 0, 0, 0, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.maximum(ima8, square(3))
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
    >>> rank.maximum(ima16, square(3))
    array([[   0,    0,    0,    0,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0,    0,    0,    0,    0]], dtype=uint16)

    """

    return _apply(_crank8.maximum, _crank16.maximum, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local mean : uint8 array or uint16 array depending on input image
        The result of the local mean.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.mean(ima8, square(3))
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
    >>> rank.mean(ima16, square(3))
    array([[1023, 1365, 2047, 1365, 1023],
           [1365, 1820, 2730, 1820, 1365],
           [2047, 2730, 4095, 2730, 2047],
           [1365, 1820, 2730, 1820, 1365],
           [1023, 1365, 2047, 1365, 1023]], dtype=uint16)

    """

    return _apply(_crank8.mean, _crank16.mean, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local meansubstraction : uint8 array or uint16 array depending on input image
        The result of the local meansubstraction.

    Examples
    --------
    to be updated
    >>> # Local meansubstraction
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.meansubstraction(ima8, square(3))
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
    >>> rank.meansubstraction(ima16, square(3))
    array([[1535, 1364, 1023, 1364, 1535],
           [1364, 3184, 2729, 3184, 1364],
           [1023, 2729, 2047, 2729, 1023],
           [1364, 3184, 2729, 3184, 1364],
           [1535, 1364, 1023, 1364, 1535]], dtype=uint16)

    """

    return _apply(
        _crank8.meansubstraction, _crank16.meansubstraction, image, selem, out=out, mask=mask,
        shift_x=shift_x, shift_y=shift_y)


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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local median : uint8 array or uint16 array depending on input image
        The result of the local median.

    Examples
    --------
    to be updated
    >>> # Local median
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 0, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.median(ima8, square(3))
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
    >>> rank.median(ima16, square(3))
    array([[   0,    0, 4095,    0,    0],
           [   0,    0, 4095,    0,    0],
           [4095, 4095, 4095, 4095, 4095],
           [   0,    0, 4095,    0,    0],
           [   0,    0, 4095,    0,    0]], dtype=uint16)

    """

    return _apply(_crank8.median, _crank16.median, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local minimum : uint8 array or uint16 array depending on input image
        The result of the local minimum.

    Examples
    --------
    to be updated
    >>> # Local minimum
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.minimum(ima8, square(3))
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
    >>> rank.minimum(ima16, square(3))
    array([[   0,    0,    0,    0,    0],
           [   0,    0,    0,    0,    0],
           [   0,    0, 4095,    0,    0],
           [   0,    0,    0,    0,    0],
           [   0,    0,    0,    0,    0]], dtype=uint16)

    """

    return _apply(_crank8.minimum, _crank16.minimum, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local modal : uint8 array or uint16 array depending on input image
        The result of the local modal.

    Examples
    --------
    to be updated
    >>> # Local modal
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 5, 6, 0],
    ...                           [0, 1, 5, 5, 0],
    ...                           [0, 0, 0, 5, 0]], dtype=np.uint8)
    >>> rank.modal(ima8, square(3))
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
    >>> rank.modal(ima16, square(3))
    array([[  0,   0,   0,   0,   0],
           [  0,   0, 100,   0,   0],
           [  0, 100, 100,   0,   0],
           [  0,   0, 500,   0,   0],
           [  0,   0, 500,   0,   0]], dtype=uint16)

    """

    return _apply(_crank8.modal, _crank16.modal, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local morph_contr_enh : uint8 array or uint16 array depending on input image
        The result of the local morph_contr_enh.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.morph_contr_enh(ima8, square(3))
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
    >>> rank.morph_contr_enh(ima16, square(3))
    array([[   0,    0,    0,    0,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0,    0,    0,    0,    0]], dtype=uint16)

    """

    return _apply(
        _crank8.morph_contr_enh, _crank16.morph_contr_enh, image, selem, out=out, mask=mask, shift_x=shift_x,
        shift_y=shift_y)


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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local pop : uint8 array or uint16 array depending on input image
        The result of the local pop.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.pop(ima8, square(3))
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
    >>> rank.pop(ima16, square(3))
    array([[4, 6, 6, 6, 4],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [4, 6, 6, 6, 4]], dtype=uint16)

    """

    return _apply(_crank8.pop, _crank16.pop, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local threshold : uint8 array or uint16 array depending on input image
        The result of the local threshold.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.threshold(ima8, square(3))
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
    >>> rank.threshold(ima16, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint16)


    """

    return _apply(
        _crank8.threshold, _crank16.threshold, image, selem, out=out, mask=mask, shift_x=shift_x,
        shift_y=shift_y)


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
    shift_x, shift_y : (int)
        Offset added to the structuring element center point.
        Shift is bounded to the structuring element sizes (center must be inside the given structuring element).

    Returns
    -------
    local tophat : uint8 array or uint16 array depending on input image
        The result of the local tophat.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.tophat(ima8, square(3))
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
    >>> rank.tophat(ima16, square(3))
    array([[4095, 4095, 4095, 4095, 4095],
           [4095,    0,    0,    0, 4095],
           [4095,    0,    0,    0, 4095],
           [4095,    0,    0,    0, 4095],
           [4095, 4095, 4095, 4095, 4095]], dtype=uint16)
    """

    return _apply(_crank8.tophat, _crank16.tophat, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)

