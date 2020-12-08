from .fused_numerics cimport np_real_numeric

cdef np_real_numeric integrate(np_real_numeric[:, ::1] sat,
                               Py_ssize_t r0, Py_ssize_t c0,
                               Py_ssize_t r1, Py_ssize_t c1) nogil
