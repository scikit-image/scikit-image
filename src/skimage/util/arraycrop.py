"""
The arraycrop module contains functions to crop values from the edges of an
n-dimensional array.
"""

import numpy as np
from numbers import Integral

__all__ = ['crop', 'bounding_box_crop']


def crop(ar, crop_width, copy=False, order='K'):
    """Crop array `ar` by `crop_width` along each dimension.

    Parameters
    ----------
    ar : array-like of rank N
        Input array.
    crop_width : {sequence, int}
        Number of values to remove from the edges of each axis.
        ``((before_1, after_1),`` ... ``(before_N, after_N))`` specifies
        unique crop widths at the start and end of each axis.
        ``((before, after),) or (before, after)`` specifies
        a fixed start and end crop for every axis.
        ``(n,)`` or ``n`` for integer ``n`` is a shortcut for
        before = after = ``n`` for all axes.
    copy : bool, optional
        If `True`, ensure the returned array is a contiguous copy. Normally,
        a crop operation will return a discontiguous view of the underlying
        input array.
    order : {'C', 'F', 'A', 'K'}, optional
        If ``copy==True``, control the memory layout of the copy. See
        ``np.copy``.

    Returns
    -------
    cropped : array
        The cropped array. If ``copy=False`` (default), this is a sliced
        view of the input array.
    """
    ar = np.array(ar, copy=False)

    if isinstance(crop_width, Integral):
        crops = [[crop_width, crop_width]] * ar.ndim
    elif isinstance(crop_width[0], Integral):
        if len(crop_width) == 1:
            crops = [[crop_width[0], crop_width[0]]] * ar.ndim
        elif len(crop_width) == 2:
            crops = [crop_width] * ar.ndim
        else:
            raise ValueError(
                f'crop_width has an invalid length: {len(crop_width)}\n'
                f'crop_width should be a sequence of N pairs, '
                f'a single pair, or a single integer'
            )
    elif len(crop_width) == 1:
        crops = [crop_width[0]] * ar.ndim
    elif len(crop_width) == ar.ndim:
        crops = crop_width
    else:
        raise ValueError(
            f'crop_width has an invalid length: {len(crop_width)}\n'
            f'crop_width should be a sequence of N pairs, '
            f'a single pair, or a single integer'
        )

    slices = tuple(slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops))
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped


def _bbox_min_max(bbox):
    """Normalize bbox=((mins...), (maxs...)) -> integer (lo, hi)."""
    try:
        lo, hi = bbox
    except Exception as e:
        raise ValueError(
            "bbox must be ((min0,...,min(D-1)), (max0,...,max(D-1)))"
        ) from e

    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if lo.ndim != 1 or hi.ndim != 1 or lo.size != hi.size:
        raise ValueError("bbox must provide equal-length 1D min and max vectors")

    lo_i = np.floor(lo).astype(int)
    hi_i = np.ceil(hi).astype(int)
    if np.any(hi_i < lo_i):
        raise ValueError("bbox has max < min along at least one axis")
    return lo_i, hi_i


def bounding_box_crop(
    image, bbox, *, channel_axis=None, clip=True, copy=False, order='K'
):
    """
    Crop ``image`` to a bounding box along spatial axes.

    Parameters
    ----------
    image : ndarray
        Input array of shape (S0, S1, ..., [C]) where [C] is an optional
        channel axis (see ``channel_axis``). Returns a *view* when possible.
    bbox : ((min0, ..., min(D-1)), (max0, ..., max(D-1)))
        Bounding box over the spatial axes only (do not include the channel axis).
        Float values are allowed; mins are floored, maxes are ceiled.
        D must equal the number of spatial axes.
    channel_axis : int or None, optional
        Index of the channel axis, or None if there is no channel axis (default : None).
        The channel axis is not cropped.
    clip : bool, optional
        If True (default), clamp bbox to array bounds. If False, raise on out-of-bounds.
    copy : bool, optional
        If True, return a contiguous copy with the selected crop.
        If False (default), return a view when possible.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout of the copy when ``copy=True``. Default is 'K'. See ``np.copy``.

    Returns
    -------
    view : ndarray
        A sliced view of ``image`` cropped to the bbox. If ``copy=True``, a contiguous
        copy is returned using the specified ``order``.

    See Also
    --------
    skimage.util.crop
        Crop by explicit widths on each axis.

    .. versionadded:: 0.26
       Added ``bounding_box_crop`` to crop N-D arrays using a spatial bounding box
       with optional ``channel_axis`` support.

    Notes
    -----
    Mins are floored to the slice start; maxes are ceiled and used as the
    exclusive slice stop (Python slicing).

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util import bounding_box_crop
    >>> im = np.arange(6*7).reshape(6, 7)
    >>> bounding_box_crop(im, ((1, 2), (5, 6))).shape
    (4, 4)
    >>> rgb = np.zeros((32, 48, 3), np.uint8)  # H, W, C
    >>> bounding_box_crop(rgb, ((4, 5), (10, 12)), channel_axis=-1).shape
    (6, 7, 3)
    >>> im = np.arange(6*7).reshape(6, 7)
    >>> bounding_box_crop(im, ((1.2, 2.1), (4.0, 5.9))).shape
    (3, 4)
    >>> chw = np.zeros((3, 32, 48), np.uint8)
    >>> bounding_box_crop(chw, ((4, 5), (10, 12)), channel_axis=0).shape
    (3, 6, 7)
    """
    arr = np.asarray(image)
    nd = arr.ndim

    if channel_axis is None:
        spatial_axes = tuple(range(nd))
    else:
        ch = channel_axis % nd
        spatial_axes = tuple(ax for ax in range(nd) if ax != ch)

    D = len(spatial_axes)
    lo, hi = _bbox_min_max(bbox)
    if lo.size != D:
        raise ValueError(
            f"bbox dimensionality (got {lo.size}) must match number of spatial axes (got {D})"
        )

    indexer = [slice(None) for _ in range(nd)]
    for k, ax in enumerate(spatial_axes):
        lo_k, hi_k = int(lo[k]), int(hi[k])
        if clip:
            lo_k = max(0, min(lo_k, arr.shape[ax]))
            hi_k = max(0, min(hi_k, arr.shape[ax]))
        if not (0 <= lo_k <= hi_k <= arr.shape[ax]):
            raise ValueError(
                f"bbox out of bounds on axis {ax}: [{lo_k}, {hi_k}] for shape {arr.shape}"
            )
        indexer[ax] = slice(lo_k, hi_k)

    result = arr[tuple(indexer)]
    return np.array(result, copy=True, order=order) if copy else result
