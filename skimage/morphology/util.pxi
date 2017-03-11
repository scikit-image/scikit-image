cimport numpy as cnp
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
@cython.unraisable_tracebacks(False)
cdef inline double euclid_dist(cnp.int32_t pt0, cnp.int32_t pt1,
                                cnp.int32_t[::1] strides):
    """Return the Euclidean distance between raveled points pt0 and pt1."""
    cdef double result = 0
    cdef double curr = 0
    for i in range(strides.shape[0]):
        curr = (pt0 // strides[i]) - (pt1 // strides[i])
        result += curr * curr
        pt0 = pt0 % strides[i]
        pt1 = pt1 % strides[i]
    return sqrt(result)
