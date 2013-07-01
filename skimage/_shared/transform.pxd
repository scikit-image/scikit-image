cimport numpy as cnp


cdef float integrate(float[:, ::1] sat, Py_ssize_t r0, Py_ssize_t c0,
                     Py_ssize_t r1, Py_ssize_t c1)
