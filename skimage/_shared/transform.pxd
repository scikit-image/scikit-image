cimport cython
cimport numpy as np


ctypedef fused integral_floating:
    cython.integral
    cython.floating


cdef integral_floating integrate(integral_floating[:, ::1] sat,
                                 Py_ssize_t r0, Py_ssize_t c0,
                                 Py_ssize_t r1, Py_ssize_t c1) nogil
