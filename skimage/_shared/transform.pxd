cimport numpy as cnp


cdef float integrate(cnp.ndarray[float, ndim=2,  mode="c"] sat,
                     Py_ssize_t r0, Py_ssize_t c0, Py_ssize_t r1, Py_ssize_t c1)
