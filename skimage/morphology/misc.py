"""Miscellaneous morphology functions."""

import numpy as np
import functools
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .._shared.utils import warn
from ._misc_cy import _remove_near_objects


# Our function names don't exactly correspond to ndimages.
# This dictionary translates from our names to scipy's.
funcs = ('erosion', 'dilation', 'opening', 'closing')
skimage2ndimage = {x: 'grey_' + x for x in funcs}

# These function names are the same in ndimage.
funcs = (
    'binary_erosion',
    'binary_dilation',
    'binary_opening',
    'binary_closing',
    'black_tophat',
    'white_tophat',
)
skimage2ndimage.update({x: x for x in funcs})


def default_footprint(func):
    """Decorator to add a default footprint to morphology functions.

    Parameters
    ----------
    func : function
        A morphology function such as erosion, dilation, opening, closing,
        white_tophat, or black_tophat.

    Returns
    -------
    func_out : function
        The function, using a default footprint of same dimension
        as the input image with connectivity 1.

    """

    @functools.wraps(func)
    def func_out(image, footprint=None, *args, **kwargs):
        if footprint is None:
            footprint = ndi.generate_binary_structure(image.ndim, 1)
        return func(image, footprint=footprint, *args, **kwargs)

    return func_out


def _check_dtype_supported(ar):
    # Should use `issubdtype` for bool below, but there's a bug in numpy 1.7
    if not (ar.dtype == bool or np.issubdtype(ar.dtype, np.integer)):
        raise TypeError(
            "Only bool or integer image types are supported. " f"Got {ar.dtype}."
        )


def remove_small_objects(ar, min_size=64, connectivity=1, *, out=None):
    """Remove objects smaller than the specified size.

    Expects ar to be an array with labeled objects, and removes objects
    smaller than min_size. If `ar` is bool, the image is first labeled.
    This leads to potentially different behavior for bool and 0-and-1
    arrays.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the objects of interest. If the array type is
        int, the ints must be non-negative.
    min_size : int, optional (default: 64)
        The smallest allowable object size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel. Used during
        labelling if `ar` is bool.
    out : ndarray
        Array of the same shape as `ar`, into which the output is
        placed. By default, a new array is created.

    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.

    See Also
    --------
    skimage.morphology.remove_near_objects

    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[0, 0, 0, 1, 0],
    ...               [1, 1, 1, 0, 0],
    ...               [1, 1, 1, 0, 1]], bool)
    >>> b = morphology.remove_small_objects(a, 6)
    >>> b
    array([[False, False, False, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> c = morphology.remove_small_objects(a, 7, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> d = morphology.remove_small_objects(a, 6, out=a)
    >>> d is a
    True

    """
    # Raising type error if not int or bool
    _check_dtype_supported(ar)

    if out is None:
        out = ar.copy()
    else:
        out[:] = ar

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        footprint = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, footprint, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    if len(component_sizes) == 2 and out.dtype != bool:
        warn(
            "Only one label was provided to `remove_small_objects`. "
            "Did you mean to use a boolean array?"
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def remove_small_holes(ar, area_threshold=64, connectivity=1, *, out=None):
    """Remove contiguous holes smaller than the specified size.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the connected components of interest.
    area_threshold : int, optional (default: 64)
        The maximum area, in pixels, of a contiguous hole that will be filled.
        Replaces `min_size`.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel.
    out : ndarray
        Array of the same shape as `ar` and bool dtype, into which the
        output is placed. By default, a new array is created.

    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small holes within connected components removed.

    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[1, 1, 1, 1, 1, 0],
    ...               [1, 1, 1, 0, 1, 0],
    ...               [1, 0, 0, 1, 1, 0],
    ...               [1, 1, 1, 1, 1, 0]], bool)
    >>> b = morphology.remove_small_holes(a, 2)
    >>> b
    array([[ True,  True,  True,  True,  True, False],
           [ True,  True,  True,  True,  True, False],
           [ True, False, False,  True,  True, False],
           [ True,  True,  True,  True,  True, False]])
    >>> c = morphology.remove_small_holes(a, 2, connectivity=2)
    >>> c
    array([[ True,  True,  True,  True,  True, False],
           [ True,  True,  True, False,  True, False],
           [ True, False, False,  True,  True, False],
           [ True,  True,  True,  True,  True, False]])
    >>> d = morphology.remove_small_holes(a, 2, out=a)
    >>> d is a
    True

    Notes
    -----
    If the array type is int, it is assumed that it contains already-labeled
    objects. The labels are not kept in the output image (this function always
    outputs a bool image). It is suggested that labeling is completed after
    using this function.

    """
    _check_dtype_supported(ar)

    # Creates warning if image is an integer image
    if ar.dtype != bool:
        warn(
            "Any labeled images will be returned as a boolean array. "
            "Did you mean to use a boolean array?",
            UserWarning,
        )

    if out is not None:
        if out.dtype != bool:
            raise TypeError("out dtype must be bool")
    else:
        out = ar.astype(bool, copy=True)

    # Creating the inverse of ar
    np.logical_not(ar, out=out)

    # removing small objects from the inverse of ar
    out = remove_small_objects(out, area_threshold, connectivity, out=out)

    np.logical_not(out, out=out)

    return out


def remove_near_objects(
    label_image,
    minimal_distance,
    *,
    priority=None,
    p_norm=2,
    out=None,
):
    """Remove objects until a minimal distance is ensured.

    Iterates over all objects (pixels that aren't zero) inside an image and
    removes "nearby" objects until all remaining ones are spaced more than a
    given minimal distance from each other.

    Parameters
    ----------
    label_image : ndarray of integers
        An n-dimensional array containing object labels, e.g. as returned by
        :func:`~.label`. A value of zero is considered background, all other
        object IDs must be positive integers.
    minimal_distance : int or float
        Remove objects with lower priority whose distance is not greater than
        this positive value.
    priority : ndarray, optional
        Defines the priority with which objects that are to close to each other
        are removed. Expects a 1-dimensional array of length
        :func:`np.amax(label_image) + 1 <numpy.amax>` that contains the priority
        for each object ID at the respective index. Objects with a lower value
        are removed first until all remaining objects fulfill the distance
        requirement. If not given, priority is given to objects with a higher
        number of samples and their ID second.
    p_norm : int or float, optional
        The Minkowski p-norm used to calculate the distance between objects.
        Defaults to 2 which corresponds to the Euclidean distance.
    out : ndarray, optional
        Array of the same shape and dtype as `image`, into which the output is
        placed. By default, a new array is created.

    Returns
    -------
    out : ndarray
        Array of the same shape as `label_image` for which objects that violate
        the `minimal_distance` condition were removed.

    See Also
    --------
    skimage.morphology.remove_small_objects

    Notes
    -----
    Setting `p_norm` to 1 will calculate the distance between objects as the
    Manhatten distance while ``np.inf`` corresponds to the Chebyshev distance.

    Constructs a kd-tree with :func:`scipy.spatial.cKDTree` of all objects
    internally.

    Examples
    --------
    >>> import skimage as ski
    >>> ski.morphology.remove_near_objects(np.array([2, 0, 1, 1]), 2)
    array([0, 0, 1, 1])
    >>> ski.morphology.remove_near_objects(
    ...     np.array([2, 0, 1, 1]), 2, priority=np.array([0, 1, 9])
    ... )
    array([2, 0, 0, 0])
    >>> label_image = np.array(
    ...     [[8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9],
    ...      [8, 8, 8, 0, 0, 0, 0, 0, 0, 9, 9],
    ...      [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
    ...      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...      [0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0],
    ...      [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ...      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ...      [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7]]
    ... )
    >>> ski.morphology.remove_near_objects(
    ...     label_image, minimal_distance=3
    ... )
    array([[8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9],
           [8, 8, 8, 0, 0, 0, 0, 0, 0, 9, 9],
           [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7]])
    """
    if minimal_distance < 0:
        raise ValueError(f"minimal_distance must be >= 0, was {minimal_distance}")
    if not np.issubdtype(label_image.dtype, np.integer):
        raise ValueError(
            f"`label_image` must be of integer dtype, got {label_image.dtype}"
        )

    if out is None:
        out = label_image.copy(order="C")
    else:
        out[:] = label_image
    # May create a copy if order is not C, account for that later
    out_raveled = out.ravel(order="C")

    if priority is None:
        priority = np.bincount(out_raveled)

    indices = np.nonzero(out_raveled)[0]
    if indices.size == 0:
        # Image with no labels, return early
        return out

    # Sort by label ID first, so that IDs of the same object are contiguous
    # in the sorted index. This allows fast discovery of the whole object by
    # simple iteration up or down the index!
    indices = indices[np.argsort(out_raveled[indices])]
    lowest_obj_id = out_raveled[indices[0]]
    if lowest_obj_id < 0:
        raise ValueError(f"found object with negative ID {lowest_obj_id!r}")
    try:
        # Sort by priority second using a stable sort to preserve the contiguous
        # sorting of objects. Because each pixel in an object has the same
        # priority we don't need to worry about separating objects.
        indices = indices[
            np.argsort(priority[out_raveled[indices]], kind="stable")[::-1]
        ]
    except IndexError as error:
        expected_shape = (np.amax(out_raveled) + 1,)
        if priority.shape != expected_shape:
            raise ValueError(
                "shape of `priority` must be (np.amax(label_image) + 1,), "
                f"expected {expected_shape}, got {priority.shape} instead"
            ) from error
        else:
            raise

    unraveled_indices = np.unravel_index(indices, label_image.shape)
    kdtree = cKDTree(
        data=np.asarray(unraveled_indices, dtype=np.float64).transpose(),
        balanced_tree=True,
    )

    _remove_near_objects(
        labels=out_raveled,
        indices=indices,
        kdtree=kdtree,
        minimal_distance=minimal_distance,
        p_norm=p_norm,
        shape=label_image.shape,
    )

    if out_raveled.base is not out:
        out[:] = out_raveled.reshape(out.shape)
    return out
