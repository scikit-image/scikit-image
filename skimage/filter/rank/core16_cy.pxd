cimport numpy as cnp


ctypedef cnp.uint16_t dtype_t


cdef dtype_t uint16_max(dtype_t a, dtype_t b)
cdef dtype_t uint16_min(dtype_t a, dtype_t b)


# 16-bit core kernel receives extra information about data bitdepth
cdef void _core16(dtype_t kernel(Py_ssize_t*, float, dtype_t,
                                 Py_ssize_t, Py_ssize_t, Py_ssize_t, float,
                                 float, Py_ssize_t, Py_ssize_t),
                  dtype_t[:, ::1] image,
                  char[:, ::1] selem,
                  char[:, ::1] mask,
                  dtype_t[:, ::1] out,
                  char shift_x, char shift_y, Py_ssize_t bitdepth,
                  float p0, float p1, Py_ssize_t s0, Py_ssize_t s1) except *
