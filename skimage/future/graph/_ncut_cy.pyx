# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
cimport numpy as cnp
import numpy as np


def cut_cost(cut, W):
    """Return the total weight of crossing edges in a bi-partition.

    Parameters
    ----------
    cut : array
        A array of booleans. Elements set to `True` belong to one
        set.
    W : array
        The weight matrix of the graph.

    Returns
    -------
    cost : float
        The total weight of crossing edges.
    """
    cdef cnp.ndarray[cnp.uint8_t, cast = True] cut_mask = np.array(cut)
    cdef Py_ssize_t num_rows, num_cols
    cdef cnp.int32_t row, col
    cdef cnp.int32_t[:] indices = W.indices
    cdef cnp.int32_t[:] indptr = W.indptr
    cdef cnp.double_t[:] data = W.data.astype(np.double)
    cdef cnp.int32_t row_index
    cdef cnp.double_t cost = 0

    num_rows = W.shape[0]
    num_cols = W.shape[1]

    for col in range(num_cols):
        for row_index in range(indptr[col], indptr[col + 1]):
            row = indices[row_index]
            if cut_mask[row] != cut_mask[col]:
                cost += data[row_index]

    return cost * 0.5
