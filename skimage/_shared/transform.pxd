cimport cython

ctypedef fused integral_floating:
    cython.integral
    cython.floating


cdef float integrate(integral_floating[:, ::1] sat,
                     Py_ssize_t r0, Py_ssize_t c0,
                     Py_ssize_t r1, Py_ssize_t c1) nogil
