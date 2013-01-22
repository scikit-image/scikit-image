cimport numpy as cnp


cdef float integrate(cnp.ndarray[float, ndim=2,  mode="c"] sat,
                     ssize_t r0, ssize_t c0, ssize_t r1, ssize_t c1)
