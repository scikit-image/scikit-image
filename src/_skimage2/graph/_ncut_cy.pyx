# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
cimport numpy as cnp
import numpy as np
cnp.import_array()

from cython cimport floating

ctypedef fused index_t:
    cnp.int32_t
    cnp.int64_t


def argmin2(cnp.float64_t[:] array):
    """Return the index of the 2nd smallest value in an array.

    Parameters
    ----------
    array : array
        The array to process.

    Returns
    -------
    min_idx2 : int
        The index of the second smallest value.
    """
    cdef cnp.float64_t min1 = np.inf
    cdef cnp.float64_t min2 = np.inf
    cdef Py_ssize_t min_idx1 = 0
    cdef Py_ssize_t min_idx2 = 0
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t n = array.shape[0]

    for i in range(n):
        x = array[i]
        if x < min1:
            min2 = min1
            min_idx2 = min_idx1
            min1 = x
            min_idx1 = i
        elif x > min1 and x < min2:
            min2 = x
            min_idx2 = i
        i += 1

    return min_idx2


def cut_cost(
    cut,
    floating[:] W_data,
    index_t[:] W_indices,
    index_t[:] W_indptr,
    int num_cols,
):
    """Return the total weight of crossing edges in a bi-partition.

    Parameters
    ----------
    cut : array
        A array of booleans. Elements set to `True` belong to one
        set.
    W_data : array
        The data of the sparse weight matrix of the graph.
    W_indices : array
        The indices of the sparse weight matrix of the graph.
    W_indptr : array
        The index pointers of the sparse weight matrix of the graph.
    num_cols : int
        The number of columns in the sparse weight matrix of the graph.

    Returns
    -------
    cost : float
        The total weight of crossing edges.
    """
    cdef cnp.ndarray[cnp.uint8_t, cast = True] cut_mask = np.array(cut)
    cdef index_t row, col
    cdef index_t row_index
    cdef cnp.float64_t cost = 0

    for col in range(num_cols):
        for row_index in range(W_indptr[col], W_indptr[col + 1]):
            row = W_indices[row_index]
            if cut_mask[row] != cut_mask[col]:
                cost += W_data[row_index]

    return cost * 0.5
