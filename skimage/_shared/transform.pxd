import cython
cimport numpy as cnp


ctypedef fused integral_floating:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cython.floating


cdef integral_floating integrate(integral_floating[:, ::1] sat,
                                 Py_ssize_t r0, Py_ssize_t c0,
                                 Py_ssize_t r1, Py_ssize_t c1) nogil
