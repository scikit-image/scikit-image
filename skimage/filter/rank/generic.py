"""The local histogram is computed using a sliding window similar to the method
described in [1]_.

Input image can be 8-bit or 16-bit, for 16-bit input images, the number of
histogram bins is determined from the maximum value present in the image.

Result image is 8-/16-bit or double with respect to the input image and the
rank filter operation.

References
----------

.. [1] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

"""

import warnings
import numpy as np
from skimage import img_as_ubyte

from . import generic_cy


__all__ = ['autolevel', 'bottomhat', 'equalize', 'gradient', 'maximum', 'mean',
           'subtract_mean', 'median', 'minimum', 'modal', 'enhance_contrast',
           'pop', 'threshold', 'tophat', 'noise_filter', 'entropy', 'otsu']


def _handle_input(image, selem, out, mask, out_dtype=None):

    if image.dtype not in (np.uint8, np.uint16):
        image = img_as_ubyte(image)

    selem = np.ascontiguousarray(img_as_ubyte(selem > 0))
    image = np.ascontiguousarray(image)

    if mask is None:
        mask = np.ones(image.shape, dtype=np.uint8)
    else:
        mask = img_as_ubyte(mask)
        mask = np.ascontiguousarray(mask)

    if out is None:
        if out_dtype is None:
            out_dtype = image.dtype
        out = np.empty_like(image, dtype=out_dtype)

    if image is out:
        raise NotImplementedError("Cannot perform rank operation in place.")

    is_8bit = image.dtype in (np.uint8, np.int8)

    if is_8bit:
        max_bin = 255
    else:
        max_bin = max(4, image.max())

    bitdepth = int(np.log2(max_bin))
    if bitdepth > 10:
        warnings.warn("Bitdepth of %d may result in bad rank filter "
                      "performance due to large number of bins." % bitdepth)

    return image, selem, out, mask, max_bin


def _apply(func, image, selem, out, mask, shift_x, shift_y, out_dtype=None):

    image, selem, out, mask, max_bin = _handle_input(image, selem, out, mask,
                                                     out_dtype)

    func(image, selem, shift_x=shift_x, shift_y=shift_y, mask=mask,
         out=out, max_bin=max_bin)

    return out


def autolevel(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Autolevel image using local histogram.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

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

    return _apply(generic_cy._autolevel, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def bottomhat(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Returns greyscale local bottomhat of an image.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    bottomhat : ndarray (same dtype as input image)
        The result of the local bottomhat.

    """

    return _apply(generic_cy._bottomhat, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def equalize(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Equalize image using local histogram.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

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

    return _apply(generic_cy._equalize, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def gradient(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local gradient of an image (i.e. local maximum - local
    minimum).


    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

    """

    return _apply(generic_cy._gradient, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def maximum(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local maximum of an image.


    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

    See also
    --------
    skimage.morphology.dilation

    Note
    ----
    * the lower algorithm complexity makes the rank.maximum() more efficient
      for larger images and structuring elements

    """

    return _apply(generic_cy._maximum, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def mean(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local mean of an image.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

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

    return _apply(generic_cy._mean, image, selem, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y)


def subtract_mean(image, selem, out=None, mask=None, shift_x=False,
                  shift_y=False):
    """Return image subtracted from its local mean.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

    """

    return _apply(generic_cy._subtract_mean, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def median(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local median of an image.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

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

    return _apply(generic_cy._median, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def minimum(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local minimum of an image.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

    See also
    --------
    skimage.morphology.erosion

    Note
    ----
    * the lower algorithm complexity makes the rank.minimum() more efficient
      for larger images and structuring elements

    """

    return _apply(generic_cy._minimum, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def modal(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local mode of an image.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

    """

    return _apply(generic_cy._modal, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def enhance_contrast(image, selem, out=None, mask=None, shift_x=False,
                     shift_y=False):
    """Enhance an image replacing each pixel by the local maximum if pixel
    greylevel is closest to maximimum than local minimum OR local minimum
    otherwise.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
        Output image.
    out : ndarray (same dtype as input image)
        The result of the local enhance_contrast.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import enhance_contrast
    >>> # Load test image
    >>> ima = data.camera()
    >>> # Local mean
    >>> avg = enhance_contrast(ima, disk(20))

    """

    return _apply(generic_cy._enhance_contrast, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def pop(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return the number (population) of pixels actually inside the
    neighborhood.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

    Examples
    --------
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> ima = 255 * np.array([[0, 0, 0, 0, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.pop(ima, square(3))
    array([[4, 6, 6, 6, 4],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [4, 6, 6, 6, 4]], dtype=uint8)

    """

    return _apply(generic_cy._pop, image, selem, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y)


def threshold(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local threshold of an image.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

    Examples
    --------
    >>> # Local threshold
    >>> from skimage.morphology import square
    >>> from skimage.filter.rank import threshold
    >>> ima = 255 * np.array([[0, 0, 0, 0, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> threshold(ima, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """

    return _apply(generic_cy._threshold, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def tophat(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local tophat of an image.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

    """

    return _apply(generic_cy._tophat, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def noise_filter(image, selem, out=None, mask=None, shift_x=False,
                 shift_y=False):
    """Returns the noise feature as described in [Hashimoto12]_

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    References
    ----------
    .. [Hashimoto12] N. Hashimoto et al. Referenceless image quality evaluation
                     for whole slide imaging. J Pathol Inform 2012;3:9.

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

    """

    # ensure that the central pixel in the structuring element is empty
    centre_r = int(selem.shape[0] / 2) + shift_y
    centre_c = int(selem.shape[1] / 2) + shift_x
    # make a local copy
    selem_cpy = selem.copy()
    selem_cpy[centre_r, centre_c] = 0

    return _apply(generic_cy._noise_filter, image, selem_cpy, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y)


def entropy(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Returns the entropy [1]_ computed locally. Entropy is computed
    using base 2 logarithm i.e. the filter returns the minimum number of
    bits needed to encode local greylevel distribution.

    Parameters
    ----------
    image : ndarray (uint8, uint16)
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (same dtype as input)
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (double)
        Output image.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Entropy_(information_theory)>

    Examples
    --------
    >>> # Local entropy
    >>> from skimage import data
    >>> from skimage.filter.rank import entropy
    >>> from skimage.morphology import disk
    >>> a8 = data.camera()
    >>> ent8 = entropy(a8, disk(5))

    """

    return _apply(generic_cy._entropy, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y,
                  out_dtype=np.double)


def otsu(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Returns the Otsu's threshold value for each pixel.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (same dtype as input image)
        Output image.

    References
    ----------
    .. [otsu] http://en.wikipedia.org/wiki/Otsu's_method

    Examples
    --------
    >>> # Local entropy
    >>> from skimage import data
    >>> from skimage.filter.rank import otsu
    >>> from skimage.morphology import disk
    >>> # defining a 8-bit test images
    >>> a8 = data.camera()
    >>> loc_otsu = otsu(a8, disk(5))
    >>> thresh_image = a8 >= loc_otsu

    """

    return _apply(generic_cy._otsu, image, selem, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y)
