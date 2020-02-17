"""
The arraycrop module contains functions to crop values from the edges of an
n-dimensional array.
"""

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
    # Since arraycrop is in the critical import path, we lazy import distutils
    # to check the version of numpy
    # After numpy 1.15, a new backward compatible function have been
    # implemented.
    # See https://github.com/numpy/numpy/pull/11966
    from distutils.version import LooseVersion as Version
    old_numpy = Version(np.__version__) < Version('1.16')
    if old_numpy:
        from numpy.lib.arraypad import _validate_lengths
    else:
        from numpy.lib.arraypad import _as_pairs

    ar = np.array(ar, copy=False)
    if old_numpy:
        crops = _validate_lengths(ar, crop_width)
    else:
        crops = _as_pairs(crop_width, ar.ndim, as_index=True)
    slices = tuple(slice(a, ar.shape[i] - b)
                   for i, (a, b) in enumerate(crops))
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped
