import cython
cimport numpy as np
from .._shared.fused_numerics cimport np_floats


@cython.boundscheck(False)
@cython.wraparound(False)
def _correlate_sparse_offsets(np_floats[:] input, Py_ssize_t[:] indices, 
                              Py_ssize_t[:] offsets, np_floats[:] values, 
                              np_floats[:] output):
    cdef Py_ssize_t i
    cdef Py_ssize_t indices_len = indices.shape[0]
    cdef Py_ssize_t value_len = values.shape[0]
    cdef Py_ssize_t off
    cdef np_floats val
    cdef Py_ssize_t vindex
    
    with nogil:
        for vindex in range(value_len):
            off = offsets[vindex]
            val = values[vindex]    
            # this loop order optimises cache access, gives 10x speedup
            for i in range(indices_len):
                output[i] += input[indices[i] + off] * val

