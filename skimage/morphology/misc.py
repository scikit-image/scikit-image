"""Miscellaneous morphology functions."""


import functools

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .._shared.utils import warn
from . import _util
from .selem import _default_selem
from ._close_objects_cy import _remove_close_objects


# Our function names don't exactly correspond to ndimages.
# This dictionary translates from our names to scipy's.
funcs = ('erosion', 'dilation', 'opening', 'closing')
skimage2ndimage = {x: 'grey_' + x for x in funcs}

# These function names are the same in ndimage.
funcs = ('binary_erosion', 'binary_dilation', 'binary_opening',
         'binary_closing', 'black_tophat', 'white_tophat')
skimage2ndimage.update({x: x for x in funcs})


def default_selem(func):
    """Decorator to add a default structuring element to morphology functions.

    Parameters
    ----------
    func : function
        A morphology function such as erosion, dilation, opening, closing,
        white_tophat, or black_tophat.

    Returns
    -------
    func_out : function
        The function, using a default structuring element of same dimension
        as the input image with connectivity 1.

    """
    @functools.wraps(func)
    def func_out(image, selem=None, *args, **kwargs):
        if selem is None:
            selem = _default_selem(image.ndim)
        return func(image, selem=selem, *args, **kwargs)

    return func_out


def _check_dtype_supported(ar):
    # Should use `issubdtype` for bool below, but there's a bug in numpy 1.7
    if not (ar.dtype == bool or np.issubdtype(ar.dtype, np.integer)):
        raise TypeError("Only bool or integer image types are supported. "
                        "Got %s." % ar.dtype)


def remove_small_objects(ar, min_size=64, connectivity=1, in_place=False):
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
        Otherwise, make a copy.

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
           [ True,  True,  True, False, False]], dtype=bool)
    >>> c = morphology.remove_small_objects(a, 7, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]], dtype=bool)
    >>> d = morphology.remove_small_objects(a, 6, in_place=True)
    >>> d is a
    True

    """
    # Raising type error if not int or bool
    _check_dtype_supported(ar)

    if in_place:
        out = ar
    else:
        out = ar.copy()

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, selem, output=ccs)
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


def remove_small_holes(ar, area_threshold=64, connectivity=1, in_place=False):
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
        If `True`, remove the connected components in the input array itself.
        Otherwise, make a copy.

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
           [ True,  True,  True,  True,  True, False]], dtype=bool)
    >>> c = morphology.remove_small_holes(a, 2, connectivity=2)
    >>> c
    array([[ True,  True,  True,  True,  True, False],
           [ True,  True,  True, False,  True, False],
           [ True, False, False,  True,  True, False],
           [ True,  True,  True,  True,  True, False]], dtype=bool)
    >>> d = morphology.remove_small_holes(a, 2, in_place=True)
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

    if in_place:
        out = ar
    else:
        out = ar.copy()

    # Creating the inverse of ar
    if in_place:
        out = np.logical_not(out, out)
    else:
        out = np.logical_not(out)

    # removing small objects from the inverse of ar
    out = remove_small_objects(out, area_threshold, connectivity, in_place)

    if in_place:
        out = np.logical_not(out, out)
    else:
        out = np.logical_not(out)

    return out


def remove_close_objects(
    image,
    minimal_distance,
    *,
    selem=None,
    connectivity=None,
    priority=None,
    inplace=None,
):
    """Remove objects until a minimal distance is ensured.

    Iterates over all objects (connected pixels that are True) inside an image
    and removes neighboring objects until all remaining ones are at least a
    minimal euclidean distance from each other.

    Parameters
    ----------
    image : ndarray
        An n-dimensional boolean array.
    minimal_distance : int or float
        The minimal allowed euclidean distance between objects. Must be
        positive.
    selem : ndarray, optional
        A structuring element used to determine the neighborhood of each
        evaluated pixel (``True`` denotes a connected pixel). It must be a
        boolean array and have the same number of dimensions as `image`. If
        neither `selem` nor `connectivity` are given, all adjacent pixels are
        considered as part of the neighborhood.
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `selem` is not None.
    priority : ndarray, optional
        An array matching `image` in shape that gives a priority for each
        object in `image`. Objects with a lower value are removed until all
        remaining objects fulfill the distance requirement. If not given,
        objects are iterated in row-major (C-style) order with decreasing
        priority.
    inplace : bool, optional
        Whether to modify `image` inplace or return a new array.

    Returns
    -------
    out : ndarray
        Array of the same shape as `image` with objects violating the distance
        condition removed.

    Notes
    -----
    This function uses an KDTree internally to efficiently find neighboring
    objects.

    Examples
    --------
    >>> from skimage.morphology import remove_close_objects
    >>> remove_close_objects(np.array([True, False, True]), 2)
    array([ True, False, False], dtype=bool)
    >>> image = np.array(
    ...     [[8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9],
    ...      [8, 8, 8, 0, 0, 0, 0, 0, 0, 9, 9],
    ...      [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
    ...      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...      [0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0],
    ...      [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ...      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ...      [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7]],
    ...     dtype=np.uint8
    ... )
    >>> result = remove_close_objects(
    ...     image.view(bool), minimal_distance=3, priority=image
    ... )
    >>> result.view(np.uint8)
    array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=uint8)
    """
    if not np.can_cast(image, bool, casting="same_kind"):
        # Cython doesn't support boolean memoryviews yet
        # https://github.com/cython/cython/issues/2204
        # and we use np.uint8_t as a workaround -> check here so we can safely
        # call `image.view(np.uint8)` before passing to Cython
        raise TypeError("image it must be a binary dtype")

    if minimal_distance < 0:
        raise ValueError(
            f"minimal_distance must be >= 0, was {minimal_distance}"
        )

    if not inplace:
        image = np.array(image, dtype=bool, order="C", copy=True)

    if image.size == 0:
        return image

    selem = _util._resolve_neighborhood(selem, connectivity, image.ndim)
    neighbor_offsets = _util._offsets_to_raveled_neighbors(
        image.shape, selem, center=((1,) * image.ndim)
    )

    labels = np.empty_like(image, dtype=np.uint32)
    ndi.label(image, selem, output=labels)

    raveled_indices = np.nonzero(image.ravel())[0]
    if raveled_indices.size == 0:
        # required, cKDTree doesn't support empty input for earlier versions
        # https://github.com/scipy/scipy/pull/10457
        return image

    if priority is not None:
        if image.shape != priority.shape:
            raise ValueError(
                "priority must have same shape as image: "
                f"{priority.shape} != {image.shape}"
            )
        sort = np.argsort(priority.ravel()[raveled_indices])[::-1]
        raveled_indices = raveled_indices[sort]

    indices = np.unravel_index(raveled_indices, image.shape)
    kdtree = cKDTree(
        data=np.asarray(indices, dtype=np.float64).T,
        balanced_tree=True,
    )

    _remove_close_objects(
        # Cython doesn't support boolean memoryviews yet
        # https://github.com/cython/cython/issues/2204
        image=image.view(np.uint8).ravel(),
        labels=labels.ravel(),
        indices=raveled_indices,
        neighbor_offsets=neighbor_offsets,
        kdtree=kdtree,
        minimal_distance=minimal_distance,
        shape=image.shape
    )
    return image
