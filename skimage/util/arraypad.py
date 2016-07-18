"""
The arraypad module contains crop, a function to remove the edges of an array.
"""
from __future__ import division, absolute_import, print_function

import numpy as np


__all__ = ['crop']


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
        ``((before, after),)`` specifies a fixed start and end crop
        for every axis.
        ``(n,)`` or ``n`` for integer ``n`` is a shortcut for
        before = after = ``n`` for all axes.
    copy : bool, optional
        Ensure the returned array is a contiguous copy. Normally, a crop
        operation will return a discontiguous view of the underlying
        input array. Passing ``copy=True`` will result in a contiguous
        copy.
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
    crops = _validate_lengths(ar, crop_width)
    slices = [slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops)]
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped
