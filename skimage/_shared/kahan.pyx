# -*- python -*-
#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np


cpdef fsum_vector(double[:] v, int calc_cumsum=0):
    """Apply Kahan summation to a vector.

    Parameters
    ----------
    v : ndarray of double
        Array to sum.
    calc_cumsum : int
        Whether or not to calculate the cumulative sum as well.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    """
    cdef:
        double s = 0
        double y, t
        double c = 0
        Py_ssize_t i
        double[:] cumsum

    if calc_cumsum:
        cumsum = np.zeros_like(v)

    for i in range(v.size):
        y = v[i] - c
        t = s + y
        c = (t - s) - y
        s = t

        if calc_cumsum:
            cumsum[i] = s

    if calc_cumsum:
        return s, np.asarray(cumsum)
    else:
        return s


def kahan_sum(arr, axis=None):
    """Calculate array sum using Kahan algorithm.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : int
        Axis along which to sum.
    """
    if axis is None:
        return fsum_vector(arr.ravel())

    out = np.apply_along_axis(fsum_vector, axis, arr)

    shape = list(arr.shape)
    shape.pop(axis)

    return out.reshape(shape)


def kahan_cumsum(arr, axis=None):
    """Calculate cumulative array sum using Kahan algorithm.

    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : int
        Axis along which to sum.
    """
    if axis is None:
        return fsum_vector(arr.ravel(), calc_cumsum=1)[1]
    else:
        out = np.apply_along_axis(lambda v: fsum_vector(v, calc_cumsum=1)[1],
                                  axis, arr)
        return out
