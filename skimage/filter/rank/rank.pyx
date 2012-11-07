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
           'modal', 'morph_contr_enh', 'pop', 'threshold', 'tophat','noise_filter']


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
    """Autolevel image using local histogram.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The result of the local autolevel.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import autolevel
    >>> # Load test image
    >>> ima = data.camera()
    >>> # Stretch image contrast locally
    >>> auto = autolevel(ima, disk(20))

    """

    return _apply(
        _crank8.autolevel, _crank16.autolevel, image, selem, out=out, mask=mask, shift_x=shift_x,
        shift_y=shift_y)


def bottomhat(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Returns greyscale local bottomhat of an image.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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


    """

    return _apply(
        _crank8.bottomhat, _crank16.bottomhat, image, selem, out=out, mask=mask, shift_x=shift_x,
        shift_y=shift_y)


def equalize(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Equalize image using local histogram.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The result of the local equalize.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import equalize
    >>> # Load test image
    >>> ima = data.camera()
    >>> # Local equalization
    >>> equ = equalize(ima, disk(20))
    """

    return _apply(
        _crank8.equalize, _crank16.equalize, image, selem, out=out, mask=mask, shift_x=shift_x,
        shift_y=shift_y)


def gradient(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local gradient of an image (i.e. local maximum - local minimum).


    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The local gradient.

    """

    return _apply(
        _crank8.gradient, _crank16.gradient, image, selem, out=out, mask=mask, shift_x=shift_x,
        shift_y=shift_y)


def maximum(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local maximum of an image.


    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The local maximum.

    See also
    --------
    skimage.morphology.dilation()

    Note
    ----
    * input image can be 8 bit or 16 bit with a value < 4096 (i.e. 12 bit)

    * the lower algorithm complexity makes the rank.maximum() more efficient for larger images and structuring elements

    """

    return _apply(_crank8.maximum, _crank16.maximum, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def mean(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local mean of an image.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The local mean.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import mean
    >>> # Load test image
    >>> ima = data.camera()
    >>> # Local mean
    >>> avg = mean(ima, disk(20))
    """

    return _apply(_crank8.mean, _crank16.mean, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def meansubstraction(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return image substracted from its local mean.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The result of the local meansubstraction.



    """

    return _apply(
        _crank8.meansubstraction, _crank16.meansubstraction, image, selem, out=out, mask=mask,
        shift_x=shift_x, shift_y=shift_y)


def median(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local median of an image.


    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The local median.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import median
    >>> # Load test image
    >>> ima = data.camera()
    >>> # Local mean
    >>> avg = median(ima, disk(20))
    """

    return _apply(_crank8.median, _crank16.median, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def minimum(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local minimum of an image.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The local minimum.

    See also
    --------
    skimage.morphology.erosion()

    Note
    ----
    * input image can be 8 bit or 16 bit with a value < 4096 (i.e. 12 bit)

    * the lower algorithm complexity makes the rank.minimum() more efficient for larger images and structuring elements

    """

    return _apply(_crank8.minimum, _crank16.minimum, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def modal(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local mode of an image.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The local modal.


    """

    return _apply(_crank8.modal, _crank16.modal, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def morph_contr_enh(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Enhance an image replacing each pixel by the local maximum if pixel graylevel is closest to maximimum
     than local minimum OR local minimum otherwise.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The result of the local morph_contr_enh.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import morph_contr_enh
    >>> # Load test image
    >>> ima = data.camera()
    >>> # Local mean
    >>> avg = morph_contr_enh(ima, disk(20))
    """

    return _apply(
        _crank8.morph_contr_enh, _crank16.morph_contr_enh, image, selem, out=out, mask=mask, shift_x=shift_x,
        shift_y=shift_y)


def pop(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return the number (population) of pixels actually inside the neighborhood.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The number of pixels belonging to the neighborhood.

    Examples
    --------
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.pop(ima, square(3))
    array([[4, 6, 6, 6, 4],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [4, 6, 6, 6, 4]], dtype=uint8)


    """

    return _apply(_crank8.pop, _crank16.pop, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def threshold(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local threshold of an image.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The result of the local threshold.

    Examples
    --------
    >>> # Local threshold
    >>> from skimage.morphology import square
    >>> from skimage.filter.rank import threshold
    >>> ima = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> threshold(ima, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)


    """

    return _apply(
        _crank8.threshold, _crank16.threshold, image, selem, out=out, mask=mask, shift_x=shift_x,
        shift_y=shift_y)


def tophat(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local tophat of an image.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The image tophat.

    """

    return _apply(_crank8.tophat, _crank16.tophat, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)

def noise_filter(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return minimum absolute diffirence between a pixel and its neighborhood

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, the algorithm uses max. 12bit histogram,
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
    out : uint8 array or uint16 array (same as input image)
        The image noise .

    """

    return _apply(_crank8.noise_filter, None, image, selem, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)