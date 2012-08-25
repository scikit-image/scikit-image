cimport numpy as cnp


cdef float integrate(cnp.ndarray[float, ndim=2,  mode="c"] sat,
                     int r0, int c0, int r1, int c1)
