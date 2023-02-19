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
funcs = ('binary_erosion', 'binary_dilation', 'binary_opening',
         'binary_closing', 'black_tophat', 'white_tophat')
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
        raise TypeError("Only bool or integer image types are supported. "
                        f"Got {ar.dtype}.")


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
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    if len(component_sizes) == 2 and out.dtype != bool:
        warn("Only one label was provided to `remove_small_objects`. "
             "Did you mean to use a boolean array?")

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
        warn("Any labeled images will be returned as a boolean array. "
             "Did you mean to use a boolean array?", UserWarning)

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


def _max_priority(label_image, *, priority_image):
    """Find maximum priority for each label.

    If the priority is given as an array with the same shape as `label_image`,
    it's not guaranteed that each pixel of an object holds the same priority
    value. This function finds the highest priority for each labeled object
    and returns it.

    Parameters
    ----------
    label_image : ndarray of integers
        An n-dimensional array of containing objects, e.g. as returned by
        :func:`~.label`. A value of zero denotes is considered background, all
        other object IDs must be positive integers.
    priority_image : ndarray
        An array with the same shape as `label_image`. The priority of each
        object is derived from the corresponding pixels. If an object's pixels
        are assigned different priorities the hightest takes precedence.

    Returns
    -------
    priority : ndarray
        A 1-dimensional array of length
        :func:`np.amax(label_image) <numpy.amax>` that contains the priority for
        each object ID at the respective index.

    Examples
    --------
    >>> label_image = np.array([[0, 1, 2], [2, 9, 9]])
    >>> priority_image = np.array([[5, 3, 1], [9, 2, 7]], dtype=np.uint8)
    >>> _max_priority(label_image, priority_image=priority_image)
    array([3, 9, 7, 0, 0, 0, 0, 0, 7], dtype=uint8)
    """
    unique_ids, inverse = np.unique(label_image.ravel(), return_inverse=True)
    max_id = unique_ids[-1]

    # New array to hold maximum priority for each label ID, fill it with the
    # smallest possible value of that dtype to ensure that real priorities
    # take precedence.
    priority = np.empty((max_id + 1,), dtype=priority_image.dtype)
    if np.issubdtype(priority.dtype, np.floating):
        min_value = np.finfo(priority.dtype).min
    else:
        min_value = np.iinfo(priority.dtype).min
    priority.fill(min_value)

    # Store max priority at the label positions corresponding to `unique_ids`.
    # Equivalent to np.maximum(priority[inverse], priority_image.ravel()) while
    # taking into account elements that are indexed by inverse more than once.
    np.maximum.at(priority, inverse, priority_image.ravel())
    # Max priority is now stored at positions corresponding to values in
    # `unique_ids`, but we want the position correspond to the index of the
    # label ID.
    priority[unique_ids] = priority[:len(unique_ids)]

    return priority


def remove_near_objects(
    label_image,
    minimal_distance,
    *,
    priority=None,
    p_norm=2,
    out=None,
):
    """Remove objects until a minimal distance is ensured.

    Iterates over all objects (connected pixels that aren't zero) inside an
    image and removes neighboring objects until all remaining ones are more than
    a minimal distance from each other.

    Parameters
    ----------
    label_image : ndarray of integers
        An n-dimensional array of containing objects, e.g. as returned by
        :func:`~.label`. A value of zero denotes is considered background, all
        other object IDs must be integers.
    minimal_distance : int or float
        Objects whose distance is not greater than this value are considered
        to close. Must be positive.
    priority : ndarray, optional
        Defines the priority with which objects that are to close to each other
        are removed. Expects a 1-dimensional array of length
        :func:`np.amax(label_image) + 1 <numpy.amax>` that contains the priority
        for each object ID at the respective index. Objects with a lower value
        are removed first until all remaining objects fulfill the distance
        requirement. If not given, priority is derived form the number of
        samples of an object.
    p_norm : int or float, optional
        The Minkowski p-norm used to calculate the distance between objects.
        Defaults to 2 which corresponds to the Euclidean distance.
    out : ndarray, optional
        Array of the same shape and dtype as `image`, into which the output is
        placed. Its memory layout must be C-contiguous. By default, a new array
        is created.

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

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html

    Examples
    --------
    >>> import skimage as ski
    >>> ski.morphology.remove_near_objects(np.array([True, False, True]), 2)
    array([False, False,  True])
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
        raise ValueError(
            f"minimal_distance must be >= 0, was {minimal_distance}"
        )
    if not np.issubdtype(label_image.dtype, np.integer):
        raise ValueError(
            f"`label_image` must be of integer dtype, got {label_image.dtype}"
        )

    if out is None:
        out = label_image.copy(order="C")
    else:
        out[:] = label_image

    if priority is None:
        priority = np.bincount(out.ravel())
        # Priority value for the background (ID 0) is not expected
        priority = priority

    # Safely ignore points that don't lie on an object's surface
    # This reduces the size of the KDTree and the number of points that
    # need to be evaluated
    # footprint = np.ones((3,) * out.ndim, dtype=out.dtype)
    # surfaces = out * footprint.sum()
    # surfaces -= ndi.convolve(out, footprint)  # edges are non-zero
    #
    # # Create index, that will define the iteration order of pixels later.
    # indices = np.nonzero(surfaces.ravel())[0]
    # del surfaces

    indices = np.nonzero(out.ravel())[0]
    # Sort by label ID first, so that IDs of the same object are contiguous
    # in the sorted index. This allows fast discovery of the whole object by
    # simple iteration up or down the index!
    indices = indices[np.argsort(out.ravel()[indices])]
    try:
        # Sort by priority second using a stable sort to preserve the contiguous
        # sorting of objects. Because each pixel in an object has the same
        # priority we don't need to worry about separating objects.
        indices = indices[
            np.argsort(priority[out.ravel()[indices]], kind="stable")[::-1]
        ]
    except IndexError as error:
        expected_shape = (np.amax(out) + 1,)
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

    # Raise error if raveling is not possible without copying
    out_raveled = out.view()
    out_raveled.shape = (out.size,)

    _remove_near_objects(
        labels=out_raveled,
        indices=indices,
        kdtree=kdtree,
        minimal_distance=minimal_distance,
        p_norm=p_norm,
        shape=label_image.shape,
    )
    return out
