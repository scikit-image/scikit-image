"""Miscellaneous morphology functions."""
import numpy as np
import functools
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .._shared.utils import warn, remove_arg
from .footprints import _default_footprint
from ._util import _resolve_neighborhood, _offsets_to_raveled_neighbors
from ._near_objects_cy import _remove_near_objects


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
            footprint = _default_footprint(image.ndim)
        return func(image, footprint=footprint, *args, **kwargs)

    return func_out


def _check_dtype_supported(ar):
    # Should use `issubdtype` for bool below, but there's a bug in numpy 1.7
    if not (ar.dtype == bool or np.issubdtype(ar.dtype, np.integer)):
        raise TypeError("Only bool or integer image types are supported. "
                        "Got %s." % ar.dtype)


@remove_arg("in_place", changed_version="1.0",
            help_msg="Please use out argument instead.")
def remove_small_objects(ar, min_size=64, connectivity=1, in_place=False,
                         *, out=None):
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
    in_place : bool, optional (default: False)
        If ``True``, remove the objects in the input array itself.
        Otherwise, make a copy. Deprecated since version 0.19. Please
        use `out` instead.
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

    if out is not None:
        in_place = False

    if in_place:
        out = ar
    else:
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


@remove_arg("in_place", changed_version="1.0",
            help_msg="Please use out argument instead.")
def remove_small_holes(ar, area_threshold=64, connectivity=1, in_place=False,
                       *, out=None):
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
    in_place : bool, optional (default: False)
        If `True`, remove the connected components in the input array
        itself. Otherwise, make a copy. Deprecated since version 0.19.
        Please use `out` instead.
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
        in_place = False

    if in_place:
        out = ar
    elif out is None:
        out = ar.astype(bool, copy=True)

    # Creating the inverse of ar
    np.logical_not(ar, out=out)

    # removing small objects from the inverse of ar
    out = remove_small_objects(out, area_threshold, connectivity, out=out)

    np.logical_not(out, out=out)

    return out


def remove_near_objects(
    image,
    minimal_distance,
    *,
    priority=None,
    p_norm=2,
    footprint=None,
    connectivity=None,
    out=None,
):
    """Remove objects until a minimal distance is ensured.

    Iterates over all objects (connected pixels that aren't zero or ``False``)
    inside an image and removes neighboring objects until all remaining ones
    are more than a minimal distance from each other.

    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    minimal_distance : int or float
        Objects whose distance is not greater than this value are considered
        to close. Must be positive.
    priority : ndarray, optional
        An array matching `image` in shape that gives a priority for each
        object in `image`. Objects with a lower value are removed first until
        all remaining objects fulfill the distance requirement. If not given,
        objects with a lower number of samples are removed first.
    p_norm : int or float, optional
        The Minkowski p-norm used to calculate the distance between objects.
        Defaults to 2 which corresponds to the Euclidean distance.
    footprint : ndarray, optional
        The footprint (structuring element) used to determine the neighborhood
        of each evaluated pixel (``True`` denotes a connected pixel). It must
        be a boolean array and have the same number of dimensions as `image`.
        If neither `footprint` nor `connectivity` are given, all adjacent
        pixels are considered as part of the neighborhood.
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    out : ndarray, optional
        Array of the same shape and dtype as `image`, into which the output is
        placed. By default, a new array is created.

    Returns
    -------
    out : ndarray
        Array of the same shape as `image` for which objects that violate the
        `minimal_distance` condition were removed.

    See Also
    --------
    skimage.morphology.remove_small_objects

    Notes
    -----
    In case the `priority` assigns the same value to different objects the
    function falls back to an object's label id as returned by
    ``scipy.ndimage.label(image, selem)`` [1]_.

    NaNs are treated as != 0 and thus are considered objects (or part of one).

    Setting `p_norm` to 1 will calculate the distance between objects as the
    Manhatten distance while ``np.inf`` corresponds to the Chebyshev distance.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html

    Examples
    --------
    >>> from skimage.morphology import remove_near_objects
    >>> remove_near_objects(np.array([True, False, True]), 2)
    array([False, False,  True])
    >>> image = np.array(
    ...     [[8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9],
    ...      [8, 8, 8, 0, 0, 0, 0, 0, 0, 9, 9],
    ...      [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
    ...      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...      [0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0],
    ...      [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ...      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ...      [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7]]
    ... )
    >>> remove_near_objects(image, minimal_distance=3, priority=image)
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
    if out is None:
        out = image.copy(order="C")

    footprint = _resolve_neighborhood(footprint, connectivity, out.ndim)
    neighbor_offsets, _ = _raveled_offsets_and_distances(
        out.shape, footprint=footprint, center=((1,) * out.ndim)
    )

    # Label objects, only samples where labels != 0 are evaluated
    labels = np.empty_like(out, dtype=np.intp)
    ndi.label(out, footprint, output=labels)

    if priority is None:
        bins = np.bincount(labels.ravel())
        priority = bins[labels.ravel()]  # Replace label id with bin count
    elif out.shape != priority.shape:
        raise ValueError(
            "priority must have same shape as image: "
            f"{priority.shape} != {out.shape}"
        )
    else:
        priority = priority.ravel()

    # Safely ignore points that don't lie on an objects surface
    # This reduces the size of the KDTree and the number of points that
    # need to be evaluated
    labels[ndi.binary_erosion(labels, structure=footprint)] = 0
    labels = labels.reshape(-1)
    raveled_indices = np.nonzero(labels)[0]

    # Use stable sort to make sorting behavior more transparent in the likely
    # event that objects have the same priority
    sort = np.argsort(priority[raveled_indices], kind="mergesort")[::-1]
    raveled_indices = raveled_indices[sort]

    indices = np.unravel_index(raveled_indices, out.shape)
    kdtree = cKDTree(
        data=np.asarray(indices, dtype=np.float64).transpose(),
        balanced_tree=True,
    )

    if np.can_cast(out, bool, casting="no"):
        # Cython doesn't support boolean memoryviews yet
        # https://github.com/cython/cython/issues/2204
        # and we use np.uint8_t as a workaround
        out = out.view(np.uint8)
        image_is_bool = True
    else:
        image_is_bool = False

    _remove_near_objects(
        image=out.reshape(-1),
        labels=labels,
        raveled_indices=raveled_indices,
        neighbor_offsets=neighbor_offsets,
        kdtree=kdtree,
        minimal_distance=minimal_distance,
        p_norm=p_norm,
        shape=out.shape,
    )

    if image_is_bool:
        out = out.base  # Restore original dtype
    return out
