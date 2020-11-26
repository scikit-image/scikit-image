"""
The arraycrop module contains functions to crop values from the edges of an
n-dimensional array.
"""

import numpy as np

__all__ = ['crop']


def crop(array, crop_width, copy=False, order='K'):
    """Crop array `array` by `crop_width` along each dimension.

    Parameters
    ----------
    array : array-like of rank N
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
    array = np.array(array, copy=False)

    if isinstance(crop_width, int):
        crops = [[crop_width, crop_width]] * array.ndim
    elif isinstance(crop_width[0], int):
        if len(crop_width) == 1:
            crops = [[crop_width[0], crop_width[0]]] * array.ndim
        elif len(crop_width) == 2:
            crops = [crop_width] * array.ndim
        else:
            raise ValueError('invalid length for sequence crop_width')
    elif len(crop_width) == 1:
        crops = [crop_width[0]] * array.ndim
    elif len(crop_width) == array.ndim:
        crops = crop_width
    else:
        raise ValueError('invalid length for sequence crop_width')

    slices = tuple(slice(a, array.shape[i] - b)
                   for i, (a, b) in enumerate(crops))
    if copy:
        cropped = np.array(array[slices], order=order, copy=True)
    else:
        cropped = array[slices]
    return cropped
