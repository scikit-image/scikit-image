"""

General Description
-------------------

These filters compute the local histogram at each pixel, using a sliding window
similar to the method described in [1]_. A histogram is built using a moving
window in order to limit redundant computation. The moving window follows a
snake-like path:

...------------------------\
/--------------------------/
\--------------------------...

The local histogram is updated at each pixel as the structuring element window
moves by, i.e. only those pixels entering and leaving the structuring element
update the local histogram. The histogram size is 8-bit (256 bins) for 8-bit
images and 2- to 16-bit for 16-bit images depending on the maximum value of the
image.

The filter is applied up to the image border, the neighborhood used is
adjusted accordingly. The user may provide a mask image (same size as input
image) where non zero values are the part of the image participating in the
histogram computation. By default the entire image is filtered.

This implementation outperforms gray.dilation for large structuring elements.

Input images will be cast in unsigned 8-bit integer or unsigned 16-bit integer
if necessary. The number of histogram bins is then determined from the maximum
value present in the image. Eventually, the output image is cast in the desired
dtype.

To do
-----

* add simple examples, adapt documentation on existing examples
* add/check existing doc
* adapting tests for each type of filter


References
----------

.. [1] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

"""

import functools
import numpy as np
from scipy import ndimage as ndi
from ...util import img_as_ubyte
from ..._shared.utils import assert_nD, warn

from . import generic_cy


__all__ = ['autolevel', 'bottomhat', 'equalize', 'gradient', 'maximum', 'mean',
           'geometric_mean', 'subtract_mean', 'median', 'minimum', 'modal',
           'enhance_contrast', 'pop', 'threshold', 'tophat', 'noise_filter',
           'entropy', 'otsu']


def _handle_input(image, selem, out=None, mask=None, out_dtype=None,
                  pixel_size=1):
    """Preprocess and verify input for filters.rank methods.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float or boolean)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    out_dtype : data-type, optional
        Desired output data-type. Default is None, which means we cast output
        in input dtype.
    pixel_size : int, optional
        Dimension of each pixel. Default value is 1.

    Returns
    -------
    image : 2-D array (np.uint8 or np.uint16)
    selem : 2-D array (np.uint8)
        The neighborhood expressed as a binary 2-D array.
    out : 3-D array (same dtype out_dtype or as input)
        Output array. The two first dimensions are the spatial ones, the third
        one is the pixel vector (length 1 by default).
    mask : 2-D array (np.uint8)
        Mask array that defines (>0) area of the image included in the local
        neighborhood.
    n_bins : int
        Number of histogram bins.
    out_dtype : data-type
        Output data-type.

    """
    assert_nD(image, 2)
    if (image.dtype in (bool, np.bool, np.bool_) or
            out_dtype in (bool, np.bool, np.bool_)):
        raise ValueError('dtype cannot be bool.')
    if image.dtype not in (np.uint8, np.uint16):
        message = ('Possible precision loss converting image of type {} to '
                   'uint8 as required by rank filters. Convert manually using '
                   'skimage.util.img_as_ubyte to silence this warning.'
                   .format(image.dtype))
        warn(message, stacklevel=5)
        image = img_as_ubyte(image)

    if out_dtype is None:
        out_dtype = image.dtype

    selem = np.ascontiguousarray(img_as_ubyte(selem > 0))
    image = np.ascontiguousarray(image)

    if mask is None:
        mask = np.ones(image.shape, dtype=np.uint8)
    else:
        mask = img_as_ubyte(mask)
        mask = np.ascontiguousarray(mask)

    if image is out:
        raise NotImplementedError("Cannot perform rank operation in place.")

    if out is None:
        out = np.empty(image.shape + (pixel_size,), dtype=out_dtype)
    else:
        if len(out.shape) == 2:
            out = out.reshape(out.shape+(pixel_size,))

    if image.dtype in (np.uint8, np.int8):
        n_bins = 256
    else:
        # Convert to a Python int to avoid the potential overflow when we add
        # 1 to the maximum of the image.
        n_bins = int(max(3, image.max())) + 1

    if n_bins > 2**10:
        warn("Bad rank filter performance is expected due to a "
             "large number of bins ({}), equivalent to an approximate "
             "bitdepth of {:.1f}.".format(n_bins, np.log2(n_bins)),
             stacklevel=2)

    return image, selem, out, mask, n_bins, out_dtype


def _apply_scalar_per_pixel(func, image, selem, out, mask, shift_x, shift_y,
                            out_dtype=None):
    """Process the specific cython function to the image.

    Parameters
    ----------
    func : function
        Cython function to apply.
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float or boolean)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float or boolean)
        If None, a new array is allocated.
    mask : ndarray (integer, float or boolean)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    out_dtype : data-type, optional
        Desired output data-type. Default is None, which means we cast output
        in input dtype.

    Returns
    -------
    out : 2-D array (same dtype as out_dtype or same as input image)
        Output image.

    """
    # preprocess and verify the input
    image, selem, out, mask, n_bins, out_dtype = _handle_input(image,
                                                               selem,
                                                               out,
                                                               mask,
                                                               out_dtype)

    # apply cython function
    func(image, selem, shift_x=shift_x, shift_y=shift_y, mask=mask,
         out=out, n_bins=n_bins)

    return out.reshape(out.shape[:2])


def _apply_vector_per_pixel(func, image, selem, out, mask, shift_x, shift_y,
                            out_dtype=None, pixel_size=1):
    """

    Parameters
    ----------
    func : function
        Cython function to apply.
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float or boolean)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float or boolean)
        If None, a new array is allocated.
    mask : ndarray (integer, float or boolean)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    out_dtype : data-type, optional
        Desired output data-type. Default is None, which means we cast output
        in input dtype.
    pixel_size : int, optional
        Dimension of each pixel.

    Returns
    -------
    out : 3-D array with float dtype of dimensions (H,W,N), where (H,W) are
        the dimensions of the input image and N is n_bins or
        ``image.max() + 1`` if no value is provided as a parameter.
        Effectively, each pixel is a N-D feature vector that is the histogram.
        The sum of the elements in the feature vector will be 1, unless no
        pixels in the window were covered by both selem and mask, in which
        case all elements will be 0.

    """
    # preprocess and verify the input
    image, selem, out, mask, n_bins, out_dtype = _handle_input(image,
                                                               selem,
                                                               out,
                                                               mask,
                                                               out_dtype,
                                                               pixel_size)

    # apply cython function
    func(image, selem, shift_x=shift_x, shift_y=shift_y, mask=mask,
         out=out, n_bins=n_bins)

    return out


def autolevel(image, selem=None, out=None, mask=None, shift_x=False,
              shift_y=False):
    """Auto-level image using local histogram.

    This filter locally stretches the histogram of gray values to cover the
    entire range of values from "white" to "black".

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    >>> from skimage.filters.rank import autolevel
    >>> img = data.camera()
    >>> auto = autolevel(img, disk(5))

    """

    return _apply_scalar_per_pixel(generic_cy._autolevel, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)


def bottomhat(image, selem=None, out=None, mask=None, shift_x=False,
              shift_y=False):
    """Local bottom-hat of an image.

    This filter computes the morphological closing of the image and then
    subtracts the result from the original image.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    >>> from skimage.filters.rank import bottomhat
    >>> img = data.camera()
    >>> out = bottomhat(img, disk(5))

    """

    return _apply_scalar_per_pixel(generic_cy._bottomhat, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)


def equalize(image, selem=None, out=None, mask=None, shift_x=False,
             shift_y=False):
    """Equalize image using local histogram.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    >>> from skimage.filters.rank import equalize
    >>> img = data.camera()
    >>> equ = equalize(img, disk(5))

    """

    return _apply_scalar_per_pixel(generic_cy._equalize, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)


def gradient(image, selem=None, out=None, mask=None, shift_x=False,
             shift_y=False):
    """Return local gradient of an image (i.e. local maximum - local minimum).

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    >>> from skimage.filters.rank import gradient
    >>> img = data.camera()
    >>> out = gradient(img, disk(5))

    """

    return _apply_scalar_per_pixel(generic_cy._gradient, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)


def maximum(image, selem=None, out=None, mask=None, shift_x=False,
            shift_y=False):
    """Return local maximum of an image.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    The lower algorithm complexity makes `skimage.filters.rank.maximum`
    more efficient for larger images and structuring elements.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filters.rank import maximum
    >>> img = data.camera()
    >>> out = maximum(img, disk(5))

    """

    return _apply_scalar_per_pixel(generic_cy._maximum, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)


def mean(image, selem=None, out=None, mask=None, shift_x=False, shift_y=False):
    """Return local mean of an image.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    >>> from skimage.filters.rank import mean
    >>> img = data.camera()
    >>> avg = mean(img, disk(5))

    """

    return _apply_scalar_per_pixel(generic_cy._mean, image, selem, out=out,
                                   mask=mask, shift_x=shift_x, shift_y=shift_y)


def geometric_mean(image, selem=None, out=None, mask=None,
                   shift_x=False, shift_y=False):
    """Return local geometric mean of an image.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    >>> from skimage.filters.rank import mean
    >>> img = data.camera()
    >>> avg = geometric_mean(img, disk(5))

    References
    ----------
    .. [1] Gonzalez, R. C. and Wood, R. E. "Digital Image Processing (3rd Edition)."
           Prentice-Hall Inc, 2006.

    """

    return _apply_scalar_per_pixel(generic_cy._geometric_mean, image, selem,
                                   out=out, mask=mask, shift_x=shift_x,
                                   shift_y=shift_y)


def subtract_mean(image, selem=None, out=None, mask=None, shift_x=False,
                  shift_y=False):
    """Return image subtracted from its local mean.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    >>> from skimage.filters.rank import subtract_mean
    >>> img = data.camera()
    >>> out = subtract_mean(img, disk(5))

    """

    return _apply_scalar_per_pixel(generic_cy._subtract_mean, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)


def median(image, selem=None, out=None, mask=None,
           shift_x=False, shift_y=False):
    """Return local median of an image.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's. If None, a
        full square of size 3 is used.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    See also
    --------
    skimage.filters.median : Implementation of a median filtering which handles
        images with floating precision.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filters.rank import median
    >>> img = data.camera()
    >>> med = median(img, disk(5))

    """

    if selem is None:
        selem = ndi.generate_binary_structure(image.ndim, image.ndim)
    return _apply_scalar_per_pixel(generic_cy._median, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)


def minimum(image, selem=None, out=None, mask=None, shift_x=False,
            shift_y=False):
    """Return local minimum of an image.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    The lower algorithm complexity makes `skimage.filters.rank.minimum` more
    efficient for larger images and structuring elements.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filters.rank import minimum
    >>> img = data.camera()
    >>> out = minimum(img, disk(5))

    """

    return _apply_scalar_per_pixel(generic_cy._minimum, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)


def modal(image, selem=None, out=None, mask=None, shift_x=False,
          shift_y=False):
    """Return local mode of an image.

    The mode is the value that appears most often in the local histogram.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    >>> from skimage.filters.rank import modal
    >>> img = data.camera()
    >>> out = modal(img, disk(5))

    """

    return _apply_scalar_per_pixel(generic_cy._modal, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)


def enhance_contrast(image, selem=None, out=None, mask=None, shift_x=False,
                     shift_y=False):
    """Enhance contrast of an image.

    This replaces each pixel by the local maximum if the pixel gray value is
    closer to the local maximum than the local minimum. Otherwise it is
    replaced by the local minimum.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filters.rank import enhance_contrast
    >>> img = data.camera()
    >>> out = enhance_contrast(img, disk(5))

    """

    return _apply_scalar_per_pixel(generic_cy._enhance_contrast, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)


def pop(image, selem=None, out=None, mask=None, shift_x=False, shift_y=False):
    """Return the local number (population) of pixels.

    The number of pixels is defined as the number of pixels which are included
    in the structuring element and the mask.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    >>> import skimage.filters.rank as rank
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

    return _apply_scalar_per_pixel(generic_cy._pop, image, selem, out=out,
                                   mask=mask, shift_x=shift_x,
                                   shift_y=shift_y)


def sum(image, selem=None, out=None, mask=None, shift_x=False, shift_y=False):
    """Return the local sum of pixels.

    Note that the sum may overflow depending on the data type of the input
    array.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    >>> import skimage.filters.rank as rank
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

    return _apply_scalar_per_pixel(generic_cy._sum, image, selem, out=out,
                                   mask=mask, shift_x=shift_x,
                                   shift_y=shift_y)


def threshold(image, selem=None, out=None, mask=None, shift_x=False,
              shift_y=False):
    """Local threshold of an image.

    The resulting binary mask is True if the gray value of the center pixel is
    greater than the local mean.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    >>> from skimage.filters.rank import threshold
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

    return _apply_scalar_per_pixel(generic_cy._threshold, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)


def tophat(image, selem=None, out=None, mask=None, shift_x=False,
           shift_y=False):
    """Local top-hat of an image.

    This filter computes the morphological opening of the image and then
    subtracts the result from the original image.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    >>> from skimage.filters.rank import tophat
    >>> img = data.camera()
    >>> out = tophat(img, disk(5))

    """

    return _apply_scalar_per_pixel(generic_cy._tophat, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)


def noise_filter(image, selem=None, out=None, mask=None, shift_x=False,
                 shift_y=False):
    """Noise feature.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    References
    ----------
    .. [1] N. Hashimoto et al. Referenceless image quality evaluation
                     for whole slide imaging. J Pathol Inform 2012;3:9.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filters.rank import noise_filter
    >>> img = data.camera()
    >>> out = noise_filter(img, disk(5))

    """

    # ensure that the central pixel in the structuring element is empty
    centre_r = int(selem.shape[0] / 2) + shift_y
    centre_c = int(selem.shape[1] / 2) + shift_x
    # make a local copy
    selem_cpy = selem.copy()
    selem_cpy[centre_r, centre_c] = 0

    return _apply_scalar_per_pixel(generic_cy._noise_filter, image, selem_cpy,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)


def entropy(image, selem=None, out=None, mask=None, shift_x=False,
            shift_y=False):
    """Local entropy.

    The entropy is computed using base 2 logarithm i.e. the filter returns the
    minimum number of bits needed to encode the local gray level
    distribution.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : ndarray (float)
        Output image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Entropy_(information_theory)

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters.rank import entropy
    >>> from skimage.morphology import disk
    >>> img = data.camera()
    >>> ent = entropy(img, disk(5))

    """

    return _apply_scalar_per_pixel(generic_cy._entropy, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y,
                                   out_dtype=np.double)


def otsu(image, selem=None, out=None, mask=None, shift_x=False, shift_y=False):
    """Local Otsu's threshold value for each pixel.

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Otsu's_method

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters.rank import otsu
    >>> from skimage.morphology import disk
    >>> img = data.camera()
    >>> local_otsu = otsu(img, disk(5))
    >>> thresh_image = img >= local_otsu

    """

    return _apply_scalar_per_pixel(generic_cy._otsu, image, selem, out=out,
                                   mask=mask, shift_x=shift_x,
                                   shift_y=shift_y)


def windowed_histogram(image, selem=None, out=None, mask=None,
                       shift_x=False, shift_y=False, n_bins=None):
    """Normalized sliding window histogram

    Parameters
    ----------
    image : 2-D array (integer, float or boolean)
        Input image.
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (integer, float, boolean or optional)
        If None, a new array is allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the structuring element center point. Shift is bounded
        to the structuring element sizes (center must be inside the given
        structuring element).
    n_bins : int or None
        The number of histogram bins. Will default to ``image.max() + 1``
        if None is passed.

    Returns
    -------
    out : 3-D array (float)
        Array of dimensions (H,W,N), where (H,W) are the dimensions of the
        input image and N is n_bins or ``image.max() + 1`` if no value is
        provided as a parameter. Effectively, each pixel is a N-D feature
        vector that is the histogram. The sum of the elements in the feature
        vector will be 1, unless no pixels in the window were covered by both
        selem and mask, in which case all elements will be 0.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters.rank import windowed_histogram
    >>> from skimage.morphology import disk
    >>> img = data.camera()
    >>> hist_img = windowed_histogram(img, disk(5))

    """

    if n_bins is None:
        n_bins = int(image.max()) + 1

    return _apply_vector_per_pixel(generic_cy._windowed_hist, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y,
                                   out_dtype=np.double,
                                   pixel_size=n_bins)


def majority(image, selem=None, out=None, mask=None, shift_x=False,
             shift_y=False):
    """Majority filter assign to each pixel the most occuring value within
    its neighborhood.

    Parameters
    ----------
    image : ndarray
        Image array (uint8, uint16 array).
    selem : 2-D array (integer, float, boolean or optional)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray (integer, float, boolean or optional)
        If None, a new array will be allocated.
    mask : ndarray (integer, float, boolean or optional)
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
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
    >>> from skimage.filters.rank import majority
    >>> from skimage.morphology import disk
    >>> img = data.camera()
    >>> maj_img = majority(img, disk(5))

    """

    return _apply_scalar_per_pixel(generic_cy._majority, image, selem,
                                   out=out, mask=mask,
                                   shift_x=shift_x, shift_y=shift_y)
