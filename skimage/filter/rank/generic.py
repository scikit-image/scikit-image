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
    """Auto-level image using local histogram.

    This filter locally stretches the histogram of greyvalues to cover the
    entire range of values from "white" to "black".

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import autolevel
    >>> img = data.camera()
    >>> auto = autolevel(img, disk(5))

    """

    return _apply(generic_cy._autolevel, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def bottomhat(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Local bottom-hat of an image.

    This filter computes the morphological closing of the image and then
    subtracts the result from the original image.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : 2-D array
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import bottomhat
    >>> img = data.camera()
    >>> out = bottomhat(img, disk(5))

    """

    return _apply(generic_cy._bottomhat, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def equalize(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Equalize image using local histogram.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import equalize
    >>> img = data.camera()
    >>> equ = equalize(img, disk(5))

    """

    return _apply(generic_cy._equalize, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def gradient(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return local gradient of an image (i.e. local maximum - local minimum).

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import gradient
    >>> img = data.camera()
    >>> out = gradient(img, disk(5))

    """

    return _apply(generic_cy._gradient, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def maximum(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return local maximum of an image.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    See also
    --------
    skimage.morphology.dilation

    Notes
    -----
    The lower algorithm complexity makes the `skimage.filter.rank.maximum`
    more efficient for larger images and structuring elements.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import maximum
    >>> img = data.camera()
    >>> out = maximum(img, disk(5))

    """

    return _apply(generic_cy._maximum, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def mean(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return local mean of an image.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import mean
    >>> img = data.camera()
    >>> avg = mean(img, disk(5))

    """

    return _apply(generic_cy._mean, image, selem, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y)


def subtract_mean(image, selem, out=None, mask=None, shift_x=False,
                  shift_y=False):
    """Return image subtracted from its local mean.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import subtract_mean
    >>> img = data.camera()
    >>> out = subtract_mean(img, disk(5))

    """

    return _apply(generic_cy._subtract_mean, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def median(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return local median of an image.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import median
    >>> img = data.camera()
    >>> med = median(img, disk(5))

    """

    return _apply(generic_cy._median, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def minimum(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return local minimum of an image.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    See also
    --------
    skimage.morphology.erosion

    Notes
    -----
    The lower algorithm complexity makes the `skimage.filter.rank.minimum` more
    efficient for larger images and structuring elements.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import minimum
    >>> img = data.camera()
    >>> out = minimum(img, disk(5))

    """

    return _apply(generic_cy._minimum, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def modal(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return local mode of an image.

    The mode is the value that appears most often in the local histogram.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import modal
    >>> img = data.camera()
    >>> out = modal(img, disk(5))

    """

    return _apply(generic_cy._modal, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def enhance_contrast(image, selem, out=None, mask=None, shift_x=False,
                     shift_y=False):
    """Enhance contrast of an image.

    This replaces each pixel by the local maximum if the pixel greyvalue is
    closer to the local maximum than the local minimum. Otherwise it is
    replaced by the local minimum.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
        Output image.
    out : 2-D array (same dtype as input image)
        The result of the local enhance_contrast.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import enhance_contrast
    >>> img = data.camera()
    >>> out = enhance_contrast(img, disk(5))

    """

    return _apply(generic_cy._enhance_contrast, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def pop(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return the local number (population) of pixels.

    The number of pixels is defined as the number of pixels which are included
    in the structuring element and the mask.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> img = 255 * np.array([[0, 0, 0, 0, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.pop(img, square(3))
    array([[4, 6, 6, 6, 4],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [4, 6, 6, 6, 4]], dtype=uint8)

    """

    return _apply(generic_cy._pop, image, selem, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y)


def sum(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return the local sum of pixels.

    Note that the sum may overflow depending on the data type of the input
    array.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage.morphology import square
    >>> import skimage.filter.rank as rank
    >>> img = np.array([[0, 0, 0, 0, 0],
    ...                 [0, 1, 1, 1, 0],
    ...                 [0, 1, 1, 1, 0],
    ...                 [0, 1, 1, 1, 0],
    ...                 [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.sum(img, square(3))
    array([[1, 2, 3, 2, 1],
           [2, 4, 6, 4, 2],
           [3, 6, 9, 6, 3],
           [2, 4, 6, 4, 2],
           [1, 2, 3, 2, 1]], dtype=uint8)

    """

    return _apply(generic_cy._sum, image, selem, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y)


def threshold(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Local threshold of an image.

    The resulting binary mask is True if the greyvalue of the center pixel is
    greater than the local mean.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage.morphology import square
    >>> from skimage.filter.rank import threshold
    >>> img = 255 * np.array([[0, 0, 0, 0, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> threshold(img, square(3))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    """

    return _apply(generic_cy._threshold, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def tophat(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Local top-hat of an image.

    This filter computes the morphological opening of the image and then
    subtracts the result from the original image.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import tophat
    >>> img = data.camera()
    >>> out = tophat(img, disk(5))

    """

    return _apply(generic_cy._tophat, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def noise_filter(image, selem, out=None, mask=None, shift_x=False,
                 shift_y=False):
    """Noise feature as described in [Hashimoto12]_.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
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
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filter.rank import noise_filter
    >>> img = data.camera()
    >>> out = noise_filter(img, disk(5))

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
    """Local entropy [1]_.

    The entropy is computed using base 2 logarithm i.e. the filter returns the
    minimum number of bits needed to encode the local greylevel distribution.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    selem : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
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
    .. [1] http://en.wikipedia.org/wiki/Entropy_(information_theory)

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filter.rank import entropy
    >>> from skimage.morphology import disk
    >>> img = data.camera()
    >>> ent = entropy(img, disk(5))

    """

    return _apply(generic_cy._entropy, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y,
                  out_dtype=np.double)


def otsu(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Local Otsu's threshold value for each pixel.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array).
    selem : 2-D array
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
    out : 2-D array (same dtype as input image)
        Output image.

    References
    ----------
    .. [otsu] http://en.wikipedia.org/wiki/Otsu's_method

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filter.rank import otsu
    >>> from skimage.morphology import disk
    >>> img = data.camera()
    >>> local_otsu = otsu(img, disk(5))
    >>> thresh_image = img >= local_otsu

    """

    return _apply(generic_cy._otsu, image, selem, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y)
