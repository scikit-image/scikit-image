cimport numpy as cnp


ctypedef cnp.uint8_t dtype_t


cdef dtype_t uint8_max(dtype_t a, dtype_t b)
cdef dtype_t uint8_min(dtype_t a, dtype_t b)


cdef dtype_t is_in_mask(Py_ssize_t rows, Py_ssize_t cols,
                        Py_ssize_t r, Py_ssize_t c,
                        char* mask)


# 8-bit core kernel receives extra information about data inferior and superior
# percentiles
cdef void _core8(dtype_t kernel(Py_ssize_t *, float, dtype_t, float,
                                float, Py_ssize_t, Py_ssize_t),
                 dtype_t[:, ::1] image,
                 char[:, ::1] selem,
                 char[:, ::1] mask,
                 dtype_t[:, ::1] out,
                 char shift_x, char shift_y, float p0, float p1,
                 Py_ssize_t s0, Py_ssize_t s1) except *
