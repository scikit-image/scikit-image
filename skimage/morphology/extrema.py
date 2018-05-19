"""extrema.py - local minima and maxima

This module provides functions to find local maxima and minima of an image.
Here, local maxima (minima) are defined as connected sets of pixels with equal
grey level which is strictly greater (smaller) than the grey level of all
pixels in direct neighborhood of the connected set. In addition, the module
provides the closely related functions h-maxima and h-minima.

Soille, P. (2003). Morphological Image Analysis: Principles and Applications
(2nd ed.), Chapter 6. Springer-Verlag New York, Inc.
"""

from __future__ import absolute_import

import numpy as np
from scipy import ndimage as ndi

from ..util import dtype_limits, invert, crop
from . import greyreconstruct
from ._extrema_cy import _local_maxima


def _add_constant_clip(image, const_value):
    """Add constant to the image while handling overflow issues gracefully.
    """
    min_dtype, max_dtype = dtype_limits(image, clip_negative=False)

    if const_value > (max_dtype - min_dtype):
        raise ValueError("The added constant is not compatible"
                         "with the image data type.")

    result = image + const_value
    result[image > max_dtype-const_value] = max_dtype
    return(result)


def _subtract_constant_clip(image, const_value):
    """Subtract constant from image while handling underflow issues.
    """
    min_dtype, max_dtype = dtype_limits(image, clip_negative=False)

    if const_value > (max_dtype-min_dtype):
        raise ValueError("The subtracted constant is not compatible"
                         "with the image data type.")

    result = image - const_value
    result[image < (const_value + min_dtype)] = min_dtype
    return(result)


def h_maxima(image, h, selem=None):
    """Determine all maxima of the image with height >= h.

    The local maxima are defined as connected sets of pixels with equal
    grey level strictly greater than the grey level of all pixels in direct
    neighborhood of the set.

    A local maximum M of height h is a local maximum for which
    there is at least one path joining M with a higher maximum on which the
    minimal value is f(M) - h (i.e. the values along the path are not
    decreasing by more than h with respect to the maximum's value) and no
    path for which the minimal value is greater.

    Parameters
    ----------
    image : ndarray
        The input image for which the maxima are to be calculated.
    h : unsigned integer
        The minimal height of all extracted maxima.
    selem : ndarray, optional
        The neighborhood expressed as an n-D array of 1's and 0's.
        Default is the ball of radius 1 according to the maximum norm
        (i.e. a 3x3 square for 2D images, a 3x3x3 cube for 3D images, etc.)

    Returns
    -------
    h_max : ndarray
       The maxima of height >= h. The result image is a binary image, where
       pixels belonging to the selected maxima take value 1, the others
       take value 0.

    See also
    --------
    skimage.morphology.extrema.h_minima
    skimage.morphology.extrema.local_maxima
    skimage.morphology.extrema.local_minima

    References
    ----------
    .. [1] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import extrema

    We create an image (quadratic function with a maximum in the center and
    4 additional constant maxima.
    The heights of the maxima are: 1, 21, 41, 61, 81, 101

    >>> w = 10
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:4,2:4] = 40; f[2:4,7:9] = 60; f[7:9,2:4] = 80; f[7:9,7:9] = 100
    >>> f = f.astype(np.int)

    We can calculate all maxima with a height of at least 40:

    >>> maxima = extrema.h_maxima(f, 40)

    The resulting image will contain 4 local maxima.
    """
    if np.issubdtype(image.dtype, np.floating):
        resolution = 2 * np.finfo(image.dtype).resolution
        if h < resolution:
            h = resolution
        h_corrected = h - resolution / 2.0
        shifted_img = image - h
    else:
        shifted_img = _subtract_constant_clip(image, h)
        h_corrected = h

    rec_img = greyreconstruct.reconstruction(shifted_img, image,
                                             method='dilation', selem=selem)
    residue_img = image - rec_img
    h_max = np.zeros(image.shape, dtype=np.uint8)
    h_max[residue_img >= h_corrected] = 1
    return h_max


def h_minima(image, h, selem=None):
    """Determine all minima of the image with depth >= h.

    The local minima are defined as connected sets of pixels with equal
    grey level strictly smaller than the grey levels of all pixels in direct
    neighborhood of the set.

    A local minimum M of depth h is a local minimum for which
    there is at least one path joining M with a deeper minimum on which the
    maximal value is f(M) + h (i.e. the values along the path are not
    increasing by more than h with respect to the minimum's value) and no
    path for which the maximal value is smaller.

    Parameters
    ----------
    image : ndarray
        The input image for which the minima are to be calculated.
    h : unsigned integer
        The minimal depth of all extracted minima.
    selem : ndarray, optional
        The neighborhood expressed as an n-D array of 1's and 0's.
        Default is the ball of radius 1 according to the maximum norm
        (i.e. a 3x3 square for 2D images, a 3x3x3 cube for 3D images, etc.)

    Returns
    -------
    h_min : ndarray
       The minima of depth >= h. The result image is a binary image, where
       pixels belonging to the selected minima take value 1, the other pixels
       take value 0.

    See also
    --------
    skimage.morphology.extrema.h_maxima
    skimage.morphology.extrema.local_maxima
    skimage.morphology.extrema.local_minima

    References
    ----------
    .. [1] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import extrema

    We create an image (quadratic function with a minimum in the center and
    4 additional constant maxima.
    The depth of the minima are: 1, 21, 41, 61, 81, 101

    >>> w = 10
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:4,2:4] = 160; f[2:4,7:9] = 140; f[7:9,2:4] = 120; f[7:9,7:9] = 100
    >>> f = f.astype(np.int)

    We can calculate all minima with a depth of at least 40:

    >>> minima = extrema.h_minima(f, 40)

    The resulting image will contain 4 local minima.
    """
    if np.issubdtype(image.dtype, np.floating):
        resolution = 2 * np.finfo(image.dtype).resolution
        if h < resolution:
            h = resolution
        h_corrected = h - resolution / 2.0
        shifted_img = image + h
    else:
        shifted_img = _add_constant_clip(image, h)
        h_corrected = h

    rec_img = greyreconstruct.reconstruction(shifted_img, image,
                                             method='erosion', selem=selem)
    residue_img = rec_img - image
    h_min = np.zeros(image.shape, dtype=np.uint8)
    h_min[residue_img >= h_corrected] = 1
    return h_min


def _set_edge_values_inplace(image, value):
    """Set edge values along all axes to a constant value.

    Parameters
    ----------
    image : ndarray
        The array to modify inplace.
    value : scalar
        The value to use. Should be compatible with `image`'s dtype.

    Examples
    --------
    >>> image = np.zeros((4, 5), dtype=int)
    >>> _set_edge_values_inplace(image, 1)
    >>> image
    array([[1, 1, 1, 1, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1],
           [1, 1, 1, 1, 1]])
    """
    for axis in range(image.ndim):
        sl = [slice(None)] * image.ndim
        # Set edge in front
        sl[axis] = 0
        image[sl] = value
        # Set edge to the end
        sl[axis] = -1
        image[sl] = value


def _fast_pad(image, value):
    """Pad an array on all axes with one constant value.

    Parameters
    ----------
    image : ndarray
        Image to pad.
    value : scalar
         The value to use. Should be compatible with `image`'s dtype.

    Returns
    -------
    padded_image : ndarray
        The new image.

    Notes
    -----
    The output of this function is equivalent to::

        np.pad(image, mode="constant", constant_values=value)

    However this method needs to only allocate and copy once which can result
    in significant speed gains if `image` is large.

    Examples
    --------
    >>> _fast_pad(np.zeros((2, 3)), 4)
    array([[4., 4., 4., 4., 4.],
           [4., 0., 0., 0., 4.],
           [4., 0., 0., 0., 4.],
           [4., 4., 4., 4., 4.]])
    """
    # Allocate padded image
    new_shape = np.array(image.shape) + 2
    new_image = np.empty(new_shape, dtype=image.dtype)

    # Copy old image into new space
    original_slice = tuple(slice(1, -1) for _ in range(image.ndim))
    new_image[original_slice] = image
    # and set the edge values
    _set_edge_values_inplace(new_image, value)

    return new_image


def _offset_to_raveled_neighbours(image_shape, selem):
    """Compute offsets to a samples neighbors if the image would be raveled.

    Parameters
    ----------
    image_shape : tuple
        The image for which the offsets are computed.
    selem : ndarray
        The neighborhood expressed as an n-D array of 1's and 0's.

    Returns
    -------
    offsets : ndarray
        Linear offsets to a samples neighbors in the raveled image.

    See Also
    --------
    skimage.morphology.watershed._compute_neighbors

    Examples
    --------
    >>> _offset_to_raveled_neighbours((4, 5), np.ones((3, 3)))
    array([-6, -5, -4, -1,  1,  4,  5,  6])
    """
    center = np.ones(selem.ndim, dtype=np.intp)
    # Center of kernel is not a neighbor
    selem[tuple(center)] = False
    connection_indices = np.transpose(np.nonzero(selem))
    offsets = (np.ravel_multi_index(connection_indices.T, image_shape) -
               np.ravel_multi_index(center.T, image_shape))
    return offsets


def local_maxima(image, selem=None, indices=False, include_border=True):
    """Find local maxima of n-dimensional array.

    The local maxima are defined as connected sets of pixels with equal gray
    level (plateaus) strictly greater than the gray levels of all pixels in the
    neighborhood.

    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    selem : int or ndarray, optional
        The structuring element used to determine the neighborhood of each
        evaluated pixel. If this is a number it is interpreted as the
        connectivity of `selem`. If an array, it must only contain 1's and 0's,
        have the same number of dimensions as `image`. If not given the maximal
        connectivity is assumed, meaning `selem` is an array of 1's.
    indices : bool, optional
        If True, the output will be an array representing indices of local
        maxima. If False, the output will be an array of 0's and 1's with the
        same shape as `image`.
    include_border : bool, optional
        If true, plateaus that touch the image border are valid maxima.

    Returns
    -------
    maxima : ndarray or tuple[ndarray]
        If `indices` is false, an array with the same shape as `image` is
        returned with 1's indicating the position of local maxima
        (0 otherwise). If `indices` is true, a tuple of one-dimensional arrays
        containing the coordinates (indices) of all found maxima.

    See Also
    --------
    skimage.morphology.local_minima
    skimage.morphology.h_maxima
    skimage.morphology.h_minima

    Notes
    -----
    This function operates on the following ideas:

    1. Make a first pass over the image's last dimension and flag candidates
       for local maxima by comparing pixels in only one direction.

    For each candidate:

    2. Perform a flood-fill to find all connected pixels that have the same
       gray value and are part of the plateau.
    3. Consider the connected neighborhood of a plateau: if any no bordering
       sample has a higher gray level, mark the plateau as a definite local
       maximum.

    Examples
    --------
    >>> from skimage.morphology import local_maxima
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3, 0] = 1
    >>> image[1:3, 4:6] = 2
    >>> image[3, 6] = 3
    >>> image
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Find local maxima by comparing to all neighboring pixels (maximal
    connectivity):

    >>> local_maxima(image)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 1]], dtype=uint8)
    >>> local_maxima(image, indices=True)
    (array([1, 1, 2, 2, 3, 3]), array([1, 2, 1, 2, 0, 6]))

    Find local maxima without comparing to diagonal pixels (connectivity 1):

    >>> local_maxima(image, selem=1)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [1, 0, 0, 0, 0, 0, 1]], dtype=uint8)

    and exclude maxima that border the image edge:

    >>> local_maxima(image, selem=1, include_border=False)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    image = np.asarray(image)

    if include_border:
        # Ensure that local maxima are always at least one smaller sample away
        # from the image border
        image = _fast_pad(image, image.min())

    if selem is None:
        selem = image.ndim
    if np.isscalar(selem):
        selem = ndi.generate_binary_structure(image.ndim, selem)
    else:
        # Validate custom structured element
        selem = np.asarray(selem, dtype=bool)
        # Must specify neighbors for all dimensions
        if selem.ndim != image.ndim:
            raise ValueError("selem and image must have the same number of "
                             "dimensions")
        # Must only specify direct neighbors
        if any(s != 3 for s in selem.shape):
            raise ValueError("dimension size in selem is not 3")

    neighbor_offsets = _offset_to_raveled_neighbours(image.shape, selem)

    # Array of flags used to store the state of each pixel during evaluation.
    # Possible states are:
    #   3 - first or last value in a dimension
    #   2 - potentially part of a maximum
    #   1 - evaluated and part of a maximum
    flags = np.zeros(image.shape, dtype=np.uint8)
    _set_edge_values_inplace(flags, value=3)

    # Up-cast dtype of image for wrapped Cython function
    if np.issubdtype(image.dtype, np.unsignedinteger):
        dtype = np.uint64
    elif np.issubdtype(image.dtype, np.integer):
        dtype = np.int64
    else:
        dtype = np.float64
    image = image.astype(dtype, order="C", copy=False)

    _local_maxima(image.ravel(), flags.ravel(), neighbor_offsets)

    if include_border:
        # Revert padding performed at the beginning of the function
        flags = crop(flags, 1)
    else:
        # No padding was performed but set edge values back to 0
        _set_edge_values_inplace(flags, value=0)

    if indices:
        return np.nonzero(flags)
    else:
        return flags


def local_minima(image, selem=None, indices=False, include_border=True):
    """Find local minima of n-dimensional array.

    The local minima are defined as connected sets of pixels with equal gray
    level (plateaus) strictly smaller than the gray levels of all pixels in the
    neighborhood.

    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    selem : int or ndarray, optional
        The structuring element used to determine the neighborhood of each
        evaluated pixel. If this is a number it is interpreted as the
        connectivity of `selem`. If an array, it must only contain 1's and 0's,
        have the same number of dimensions as `image`. If not given the maximal
        connectivity is assumed, meaning `selem` is an array of 1's.
    indices : bool, optional
        If True, the output will be an array representing indices of local
        minima. If False, the output will be an array of 0's and 1's with the
        same shape as `image`.
    include_border : bool, optional
        If true, plateaus that touch the image border are valid minima.

    Returns
    -------
    minima : ndarray or tuple[ndarray]
        If `indices` is false, an array with the same shape as `image` is
        returned with 1's indicating the position of local minima
        (0 otherwise). If `indices` is true, a tuple of one-dimensional arrays
        containing the coordinates (indices) of all found minima.

    See Also
    --------
    skimage.morphology.local_maxima
    skimage.morphology.h_maxima
    skimage.morphology.h_minima

    Notes
    -----
    This function operates on the following ideas:

    1. Make a first pass over the image's last dimension and flag candidates
       for local minima by comparing pixels in only one direction.

    For each candidate:

    2. Perform a flood-fill to find all connected pixels that have the same
       gray value and are part of the plateau.
    3. Consider the connected neighborhood of a plateau: if any no bordering
       sample has a smaller gray level, mark the plateau as a definite local
       minimum.

    Examples
    --------
    >>> from skimage.morphology import local_minima
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = -1
    >>> image[3, 0] = -1
    >>> image[1:3, 4:6] = -2
    >>> image[3, 6] = -3
    >>> image
    array([[ 0,  0,  0,  0,  0,  0,  0],
           [ 0, -1, -1,  0, -2, -2,  0],
           [ 0, -1, -1,  0, -2, -2,  0],
           [-1,  0,  0,  0,  0,  0, -3]])

    Find local minima by comparing to all neighboring pixels (maximal
    connectivity):

    >>> local_minima(image)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 1]], dtype=uint8)
    >>> local_minima(image, indices=True)
    (array([1, 1, 2, 2, 3, 3]), array([1, 2, 1, 2, 0, 6]))

    Find local minima without comparing to diagonal pixels (connectivity 1):

    >>> local_minima(image, selem=1)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [1, 0, 0, 0, 0, 0, 1]], dtype=uint8)

    and exclude minima that border the image edge:

    >>> local_minima(image, selem=1, include_border=False)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    return local_maxima(invert(image), selem, indices, include_border)
