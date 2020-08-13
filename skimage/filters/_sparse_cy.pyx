cimport numpy as np

ctypedef np.int_t DTYPE_INT_T
ctypedef np.float_t DTYPE_FLOAT_T
def _correlate_sparse_offsets(double[:] input, long long [:] indices, 
                                long long [:] offsets, double[:] values, 
                                double[:] output):
    cdef int i, j
    cdef int indices_len = indices.shape[0]
    cdef int value_len = values.shape[0]
    cdef int off
    cdef float val
    cdef int vindex
    
    for vindex in range(value_len):
        off = offsets[vindex]
        val = values[vindex]
        # this loop order optimises cache access, gives 10x speedup
        for i in range(indices_len):
            output[i] += input[indices[i] + off] * val

