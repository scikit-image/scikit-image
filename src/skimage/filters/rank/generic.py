"""


General Description
-------------------

These filters compute the local histogram at each pixel, using a sliding window
similar to the method described in [1]_. A histogram is built using a moving
window in order to limit redundant computation. The moving window follows a
snake-like path:

...------------------------↘
↙--------------------------↙
↘--------------------------...

The local histogram is updated at each pixel as the footprint window
moves by, i.e. only those pixels entering and leaving the footprint
update the local histogram. The histogram size is 8-bit (256 bins) for 8-bit
images and 2- to 16-bit for 16-bit images depending on the maximum value of the
image.

The filter is applied up to the image border, the neighborhood used is
adjusted accordingly. The user may provide a mask image (same size as input
image) where non zero values are the part of the image participating in the
histogram computation. By default the entire image is filtered.

This implementation outperforms :func:`skimage.morphology.dilation`
for large footprints.

Input images will be cast in unsigned 8-bit integer or unsigned 16-bit integer
if necessary. The number of histogram bins is then determined from the maximum
value present in the image. Eventually, the output image is cast in the input
dtype, or the `output_dtype` if set.

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

from _skimage2.filters.rank.generic import (
    autolevel as _ski2_autolevel,
    equalize as _ski2_equalize,
    gradient as _ski2_gradient,
    maximum as _ski2_maximum,
    mean as _ski2_mean,
    geometric_mean as _ski2_geometric_mean,
    subtract_mean as _ski2_subtract_mean,
    median as _ski2_median,
    minimum as _ski2_minimum,
    modal as _ski2_modal,
    enhance_contrast as _ski2_enhance_contrast,
    pop as _ski2_pop,
    threshold as _ski2_threshold,
    noise_filter as _ski2_noise_filter,
    entropy as _ski2_entropy,
    otsu as _ski2_otsu,
    majority as _ski2_majority,
    sum as _ski2_sum,
    windowed_histogram as _ski2_windowed_histogram,
)

__all__ = [
    'autolevel',
    'equalize',
    'gradient',
    'maximum',
    'mean',
    'geometric_mean',
    'subtract_mean',
    'median',
    'minimum',
    'modal',
    'enhance_contrast',
    'pop',
    'threshold',
    'noise_filter',
    'entropy',
    'otsu',
]


def autolevel(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Auto-level image using local histogram.

    This filter locally stretches the histogram of gray values to cover the
    entire range of values from "white" to "black".

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import autolevel
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> auto = autolevel(img, disk(5))
    >>> auto_vol = autolevel(volume, ball(5))
    """
    return _ski2_autolevel(
        image=image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def equalize(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Equalize image using local histogram.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray, same dtype as `image`
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import equalize
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> equ = equalize(img, disk(5))
    >>> equ_vol = equalize(volume, ball(5))
    """
    return _ski2_equalize(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def gradient(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Return local gradient of an image (i.e. local maximum - local minimum).

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray, same dtype as `image`
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import gradient
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = gradient(img, disk(5))
    >>> out_vol = gradient(volume, ball(5))
    """
    return _ski2_gradient(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def maximum(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Return local maximum of an image.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray, same dtype as `image`
        Output image.

    See also
    --------
    skimage.morphology.dilation

    Notes
    -----
    The lower algorithm complexity makes `skimage.filters.rank.maximum`
    more efficient for larger images and footprints.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import maximum
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = maximum(img, disk(5))
    >>> out_vol = maximum(volume, ball(5))
    """
    return _ski2_maximum(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def mean(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Return local mean of an image.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray, same dtype as `image`
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import mean
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> avg = mean(img, disk(5))
    >>> avg_vol = mean(volume, ball(5))
    """
    return _ski2_mean(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def geometric_mean(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0
):
    """Return local geometric mean of an image.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray, same dtype as `image`
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import mean
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> avg = geometric_mean(img, disk(5))
    >>> avg_vol = geometric_mean(volume, ball(5))

    References
    ----------
    .. [1] Gonzalez, R. C. and Woods, R. E. "Digital Image Processing
           (3rd Edition)." Prentice-Hall Inc, 2006.
    """
    return _ski2_geometric_mean(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def subtract_mean(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0
):
    """Return image subtracted from its local mean.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray, same dtype as `image`
        Output image.

    Notes
    -----
    Subtracting the mean value may introduce underflow. To compensate
    this potential underflow, the obtained difference is downscaled by
    a factor of 2 and shifted by `n_bins / 2 - 1`, the median value of
    the local histogram (`n_bins = max(3, image.max()) +1` for 16-bits
    images and 256 otherwise).

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import subtract_mean
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = subtract_mean(img, disk(5))
    >>> out_vol = subtract_mean(volume, ball(5))
    """
    return _ski2_subtract_mean(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def median(image, footprint=None, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Return local median of an image.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's. If None, a
        full square of size 3 is used.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray, same dtype as `image`
        Output image.

    See also
    --------
    skimage.filters.median : Implementation of a median filtering which handles
        images with floating precision.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import median
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> med = median(img, disk(5))
    >>> med_vol = median(volume, ball(5))
    """
    return _ski2_median(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def minimum(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Return local minimum of an image.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        Output image.

    See also
    --------
    skimage.morphology.erosion

    Notes
    -----
    The lower algorithm complexity makes `skimage.filters.rank.minimum` more
    efficient for larger images and footprints.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import minimum
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = minimum(img, disk(5))
    >>> out_vol = minimum(volume, ball(5))
    """
    return _ski2_minimum(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def modal(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Return local mode of an image.

    The mode is the value that appears most often in the local histogram.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray, same dtype as `image`
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import modal
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = modal(img, disk(5))
    >>> out_vol = modal(volume, ball(5))
    """
    return _ski2_modal(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def enhance_contrast(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0
):
    """Enhance contrast of an image.

    This replaces each pixel by the local maximum if the pixel gray value is
    closer to the local maximum than the local minimum. Otherwise it is
    replaced by the local minimum.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ([P,] M, N) ndarray, same dtype as `image`
        Output image

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import enhance_contrast
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = enhance_contrast(img, disk(5))
    >>> out_vol = enhance_contrast(volume, ball(5))
    """
    return _ski2_enhance_contrast(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def pop(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Return the local number (population) of pixels.

    The number of pixels is defined as the number of pixels which are included
    in the footprint and the mask.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        Output image.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import footprint_rectangle # Need to add 3D example
    >>> import skimage.filters.rank as rank
    >>> img = 255 * np.array([[0, 0, 0, 0, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.pop(img, footprint_rectangle((3, 3)))
    array([[4, 6, 6, 6, 4],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [4, 6, 6, 6, 4]], dtype=uint8)
    """
    return _ski2_pop(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def sum(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Return the local sum of pixels.

    Note that the sum may overflow depending on the data type of the input
    array.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        Output image.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import footprint_rectangle # Need to add 3D example
    >>> import skimage.filters.rank as rank         # Cube seems to fail but
    >>> img = np.array([[0, 0, 0, 0, 0],            # Ball can pass
    ...                 [0, 1, 1, 1, 0],
    ...                 [0, 1, 1, 1, 0],
    ...                 [0, 1, 1, 1, 0],
    ...                 [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> rank.sum(img, footprint_rectangle((3, 3)))
    array([[1, 2, 3, 2, 1],
           [2, 4, 6, 4, 2],
           [3, 6, 9, 6, 3],
           [2, 4, 6, 4, 2],
           [1, 2, 3, 2, 1]], dtype=uint8)
    """
    return _ski2_sum(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def threshold(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Local threshold of an image.

    The resulting binary mask is True if the gray value of the center pixel is
    greater than the local mean.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        Output image.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import footprint_rectangle # Need to add 3D example
    >>> from skimage.filters.rank import threshold
    >>> img = 255 * np.array([[0, 0, 0, 0, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 1, 1, 1, 0],
    ...                       [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> threshold(img, footprint_rectangle((3, 3)))
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)
    """
    return _ski2_threshold(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def noise_filter(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0
):
    """Noise feature.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    References
    ----------
    .. [1] N. Hashimoto et al. Referenceless image quality evaluation
                     for whole slide imaging. J Pathol Inform 2012;3:9.

    Returns
    -------
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        Output image.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk, ball
    >>> from skimage.filters.rank import noise_filter
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> out = noise_filter(img, disk(5))
    >>> out_vol = noise_filter(volume, ball(5))
    """
    return _ski2_noise_filter(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def entropy(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Local entropy.

    The entropy is computed using base 2 logarithm i.e. the filter returns the
    minimum number of bits needed to encode the local gray level
    distribution.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ndarray of shape ([P,] M, N) and dtype float
        Output image.

    References
    ----------
    .. [1] `https://en.wikipedia.org/wiki/Entropy_(information_theory) <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters.rank import entropy
    >>> from skimage.morphology import disk, ball
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> ent = entropy(img, disk(5))
    >>> ent_vol = entropy(volume, ball(5))
    """
    return _ski2_entropy(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def otsu(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Local Otsu's threshold value for each pixel.

    Parameters
    ----------
    image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
        Input image.
    footprint : ndarray
        The neighborhood expressed as an ndarray of 1's and 0's.
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ndarray of shape ([P,] M, N), same dtype as input `image`
        Output image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Otsu's_method

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters.rank import otsu
    >>> from skimage.morphology import disk, ball
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> local_otsu = otsu(img, disk(5))
    >>> thresh_image = img >= local_otsu
    >>> local_otsu_vol = otsu(volume, ball(5))
    >>> thresh_image_vol = volume >= local_otsu_vol
    """
    return _ski2_otsu(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )


def windowed_histogram(
    image, footprint, out=None, mask=None, shift_x=0, shift_y=0, n_bins=None
):
    """Compute normalized sliding window histogram.

    Parameters
    ----------
    image : ndarray of shape (H, W) and dtype (int or float)
        Input image.
    footprint : ndarray of dtype (int or float)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray of shape (H, W, N) and dtype (int or float), optional
        If None, a new array is allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    n_bins : int or None
        The number of histogram bins. Defaults to ``image.max() + 1``
        if None is passed.

    Returns
    -------
    out : ndarray of shape (H, W, N) and dtype float
        `N` is `n_bins` or ``image.max() + 1`` if no value is passed to
        `n_bins`. Effectively, each pixel is a N-D feature
        vector that is the histogram. The sum of the elements in the feature
        vector is 1, unless no pixels in the window were covered by both
        `footprint` and `mask`, in which case all elements are 0.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters.rank import windowed_histogram
    >>> from skimage.morphology import disk, ball
    >>> import numpy as np
    >>> img = data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10,10,10), dtype=np.uint8)
    >>> hist_img = windowed_histogram(img, disk(5))
    """
    return _ski2_windowed_histogram(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        n_bins=n_bins,
    )


def majority(image, footprint, *, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
    """Assign to each pixel the most common value within its neighborhood.

    Parameters
    ----------
    image : ndarray of dtype (int or float)
        Image array.
    footprint : 2-D array (integer or float)
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray of dtype int, optional
        If None, a new array will be allocated.
    mask : ndarray of dtype (int or float), optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y, shift_z : int, optional
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).

    Returns
    -------
    out : ndarray of dtype int, optional
        Output image.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski
    >>> img = ski.data.camera()
    >>> rng = np.random.default_rng()
    >>> volume = rng.integers(0, 255, size=(10, 10, 10), dtype=np.uint8)
    >>> maj_img = ski.filters.rank.majority(img, ski.morphology.disk(5))
    >>> maj_img_vol = ski.filters.rank.majority(volume, ski.morphology.ball(5))
    """
    return _ski2_majority(
        image,
        footprint=footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
    )
