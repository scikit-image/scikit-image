"""The local histogram is computed using a sliding window similar to the method
described in [1]_.

Input image can be 8-bit or 16-bit, for 16-bit input images, the number of
histogram bins is determined from the maximum value present in the image.

Result image is 8- or 16-bit with respect to the input image.

References
----------

.. [1] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

"""

import numpy as np
from skimage import img_as_ubyte, img_as_uint
from ... import get_log
log = get_log()

from . import generic8_cy, generic16_cy


__all__ = ['autolevel', 'bottomhat', 'equalize', 'gradient', 'maximum', 'mean',
           'meansubtraction', 'median', 'minimum', 'modal', 'morph_contr_enh',
           'pop', 'threshold', 'tophat', 'noise_filter', 'entropy', 'otsu']


import numpy as np


def find_bitdepth(image):
    """returns the max bith depth of a uint16 image
    """
    umax = np.max(image)
    if umax > 2:
        return int(np.log2(umax))
    else:
        return 1


def _apply(func8, func16, image, selem, out, mask, shift_x, shift_y):
    selem = img_as_ubyte(selem > 0)
    image = np.ascontiguousarray(image)

    if mask is None:
        mask = np.ones(image.shape, dtype=np.uint8)
    else:
        mask = np.ascontiguousarray(mask)
        mask = img_as_ubyte(mask)

    if image is out:
        raise NotImplementedError("Cannot perform rank operation in place.")

    is_8bit = image.dtype in (np.uint8, np.int8)

    if func8 is not None and (is_8bit or func16 is None):
        out = _apply8(func8, image, selem, out, mask, shift_x, shift_y)
    else:
        image = img_as_uint(image)
        if out is None:
            out = np.zeros(image.shape, dtype=np.uint16)
        bitdepth = find_bitdepth(image)
        if bitdepth > 10:
            log.warn("Bitdepth of %d may result in bad rank filter "
                     "performance." % bitdepth)
        func16(image, selem, shift_x=shift_x, shift_y=shift_y, mask=mask,
               bitdepth=bitdepth + 1, out=out)

    return out


def _apply8(func8, image, selem, out, mask, shift_x, shift_y):
    if out is None:
        out = np.zeros(image.shape, dtype=np.uint8)
    image = img_as_ubyte(image)
    func8(image, selem, shift_x=shift_x, shift_y=shift_y,
          mask=mask, out=out)
    return out


def autolevel(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Autolevel image using local histogram.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

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

    return _apply(generic8_cy.autolevel, generic16_cy.autolevel, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def bottomhat(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Returns greyscale local bottomhat of an image.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    local bottomhat : uint8 array or uint16 array depending on input image
        The result of the local bottomhat.

    """

    return _apply(generic8_cy.bottomhat, generic16_cy.bottomhat, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def equalize(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Equalize image using local histogram.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

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

    return _apply(generic8_cy.equalize, generic16_cy.equalize, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def gradient(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local gradient of an image (i.e. local maximum - local
    minimum).


    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : uint8 array or uint16 array (same as input image)
        The local gradient.

    """

    return _apply(generic8_cy.gradient, generic16_cy.gradient, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def maximum(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local maximum of an image.


    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : uint8 array or uint16 array (same as input image)
        The local maximum.

    See also
    --------
    skimage.morphology.dilation

    Note
    ----
    * input image can be 8-bit or 16-bit with a value < 4096 (i.e. 12 bit)
    * the lower algorithm complexity makes the rank.maximum() more efficient for
      larger images and structuring elements

    """

    return _apply(generic8_cy.maximum, generic16_cy.maximum, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def mean(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local mean of an image.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

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

    return _apply(generic8_cy.mean, generic16_cy.mean, image, selem, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y)


def meansubtraction(image, selem, out=None, mask=None, shift_x=False,
                     shift_y=False):
    """Return image subtracted from its local mean.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : uint8 array or uint16 array (same as input image)
        The result of the local meansubtraction.

    """

    return _apply(generic8_cy.meansubtraction, generic16_cy.meansubtraction,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y)


def median(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local median of an image.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

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

    return _apply(generic8_cy.median, generic16_cy.median, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def minimum(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local minimum of an image.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : uint8 array or uint16 array (same as input image)
        The local minimum.

    See also
    --------
    skimage.morphology.erosion

    Note
    ----
    * input image can be 8-bit or 16-bit with a value < 4096 (i.e. 12 bit)
    * the lower algorithm complexity makes the rank.minimum() more efficient
      for larger images and structuring elements

    """

    return _apply(generic8_cy.minimum, generic16_cy.minimum, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def modal(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local mode of an image.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : uint8 array or uint16 array (same as input image)
        The local modal.

    """

    return _apply(generic8_cy.modal, generic16_cy.modal, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def morph_contr_enh(image, selem, out=None, mask=None, shift_x=False,
                    shift_y=False):
    """Enhance an image replacing each pixel by the local maximum if pixel
    greylevel is closest to maximimum than local minimum OR local minimum
    otherwise.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

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

    return _apply(generic8_cy.morph_contr_enh, generic16_cy.morph_contr_enh,
                  image, selem, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y)


def pop(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return the number (population) of pixels actually inside the
    neighborhood.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : uint8 array or uint16 array (same as input image)
        The number of pixels belonging to the neighborhood.

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

    return _apply(generic8_cy.pop, generic16_cy.pop, image, selem, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y)


def threshold(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local threshold of an image.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : uint8 array or uint16 array (same as input image)
        The result of the local threshold.

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

    return _apply(generic8_cy.threshold, generic16_cy.threshold, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def tophat(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Return greyscale local tophat of an image.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : uint8 array or uint16 array (same as input image)
        The image tophat.

    """

    return _apply(generic8_cy.tophat, generic16_cy.tophat, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


def noise_filter(image, selem, out=None, mask=None, shift_x=False,
                 shift_y=False):
    """Returns the noise feature as described in [Hashimoto12]_

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
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
    out : uint8 array or uint16 array (same as input image)
        The image noise.

    """

    # ensure that the central pixel in the structuring element is empty
    centre_r = int(selem.shape[0] / 2) + shift_y
    centre_c = int(selem.shape[1] / 2) + shift_x
    # make a local copy
    selem_cpy = selem.copy()
    selem_cpy[centre_r, centre_c] = 0

    return _apply(generic8_cy.noise_filter, None, image, selem_cpy, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y)


def entropy(image, selem, out=None, mask=None, shift_x=False, shift_y=False):
    """Returns the entropy [1]_ computed locally. Entropy is computed
    using base 2 logarithm i.e. the filter returns the minimum number of
    bits needed to encode local greylevel distribution.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16).
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        If None, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : uint8 array or uint16 array (same as input image)
        Entropy x10 (uint8 images) and entropy x1000 (uint16 images)

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Entropy_(information_theory)

    Examples
    --------
    >>> # Local entropy
    >>> from skimage import data
    >>> from skimage.filter.rank import entropy
    >>> from skimage.morphology import disk
    >>> # defining a 8- and a 16-bit test images
    >>> a8 = data.camera()
    >>> a16 = data.camera().astype(np.uint16) * 4
    >>> # pixel values contain 10x the local entropy
    >>> ent8 = entropy(a8, disk(5))
    >>> # pixel values contain 1000x the local entropy
    >>> ent16 = entropy(a16, disk(5))

    """

    return _apply(generic8_cy.entropy, generic16_cy.entropy, image, selem,
                  out=out, mask=mask, shift_x=shift_x, shift_y=shift_y)


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
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : uint8 array
        Otsu's threshold values

    References
    ----------
    .. [otsu] http://en.wikipedia.org/wiki/Otsu's_method

    Notes
    -----
    * input image are 8-bit only

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

    return _apply(generic8_cy.otsu, None, image, selem, out=out,
                  mask=mask, shift_x=shift_x, shift_y=shift_y)
