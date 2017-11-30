# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
cimport numpy as cnp
import numpy as np

def find_cut(ind_sorted_py, ind_sort_py, W):
    """
    Calculates cost for all valid bipartitions and returns the index
    corresponding to the best one.

    A valid bipartition is one that splits the indices specified
    by ind_sorted_py in two.

    Parameters
    ----------
    ind_sorted_py : array
        Indices of the partition vector in sorted order
    ind_sort_py : array
        Maps unsorted indices to their index in sorted order
    W : array
        The weight matrix of the graph sorted according to the entries
        in the partition vector.

    Returns
    -------
    min_cut : int
        Normalized cut value for the best valid partition.
    min_col : int
        Final index of ind_sorted corresponding to subgraph A
    """
    cdef Py_ssize_t num_cols
    cdef cnp.int32_t col, min_col, row_index
    cdef cnp.int32_t[:] ind_sorted = ind_sorted_py.astype(np.int32)
    cdef cnp.int32_t[:] ind_sort = ind_sort_py.astype(np.int32)
    cdef cnp.int32_t[:] indices = W.indices
    cdef cnp.int32_t[:] indptr = W.indptr
    cdef cnp.double_t[:] data = W.data.astype(np.double)
    cdef cnp.double_t cut, min_cut
    cdef cnp.double_t assoc_av, assoc_vv, assoc_ab
    cdef cnp.double_t less_sum, eq, more_sum

    # Notation taken from Shi & Malik 2000
    # At start assoc(A,V) is sum of all weights ie. assoc(V,V)
    # assoc(A,B) equivalent to cut(A,B)
    assoc_vv = W.data.sum()
    assoc_av = assoc_vv
    assoc_ab = 0.0
    num_cols = W.shape[0]
    min_col = -1
    min_cut = 2.0

    # Final index skipped because it's equivalent to not partitioning
    for col in ind_sorted[:num_cols-1]:
        less_sum = 0.0
        more_sum = 0.0
        for row_index in range(indptr[col], indptr[col+1]):
            row = indices[row_index]
            if ind_sort[row] < ind_sort[col]:
                less_sum += data[row_index]
            elif ind_sort[row] == ind_sort[col]:
                eq = data[row_index]
            else:
                more_sum += data[row_index]
        assoc_ab += more_sum - less_sum
        assoc_av -= more_sum + eq + less_sum

        # The normalized cuts value.
        # Uses identity:
        # assoc(B,V) = assoc(V,V) - assoc(A,V)
        cut = assoc_ab/assoc_av + assoc_ab/(assoc_vv - assoc_av)
        if cut < min_cut:
            min_cut = cut
            min_col = ind_sort[col]
    return min_cut, min_col
