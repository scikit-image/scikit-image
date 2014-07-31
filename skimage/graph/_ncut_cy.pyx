# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
cimport numpy as cnp
import numpy as np


def argmin2(cnp.float64_t[:] array):
    """Return the index of the 2nd smallest value in an array.

    Parameters
    ----------
    a : array
        The array to process.

    Returns
    -------
    i : int
        The index of the second smallest value.
    """
    cdef cnp.float64_t min1 = np.inf
    cdef cnp.float64_t min2 = np.inf
    cdef Py_ssize_t i1 = 0
    cdef Py_ssize_t i2 = 0
    cdef Py_ssize_t i = 0

    while i < array.shape[0]:
        x = array[i]
        if x < min1:
            min2 = min1
            i2 = i1
            min1 = x
            i1 = i
        elif x > min1 and x < min2:
            min2 = x
            i2 = i
        i += 1

    return i2


def cut_cost(mask, W):
    mask = np.array(mask)

    cdef Py_ssize_t num_rows, num_cols
    cdef cnp.int32_t row, col
    cdef cnp.int32_t[:] indices = W.indices
    cdef cnp.int32_t[:] indptr = W.indptr
    cdef cnp.float64_t[:] data = W.data
    cdef cnp.int32_t row_index
    cdef cnp.double_t cost = 0

    num_rows = W.shape[0]
    num_cols = W.shape[1]

    col = 0
    while col < num_cols:
        row_index = indptr[col]
        while row_index < indptr[col+1]:
            row = indices[row_index]
            if mask[row] != mask[col]:
                cost += <cnp.double_t>data[row_index]
            row_index += 1
        col += 1

    return cost*0.5
