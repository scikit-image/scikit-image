cimport numpy as cnp


ctypedef cnp.uint8_t dtype_t


cdef dtype_t uint8_max(dtype_t a, dtype_t b)
cdef dtype_t uint8_min(dtype_t a, dtype_t b)


cdef dtype_t is_in_mask(Py_ssize_t rows, Py_ssize_t cols,
                        Py_ssize_t r, Py_ssize_t c,
                        dtype_t * mask)


# 8-bit core kernel receives extra information about data inferior and superior
# percentiles
cdef void _core8(dtype_t kernel(Py_ssize_t *, float, dtype_t, float,
                                float, Py_ssize_t, Py_ssize_t),
                 cnp.ndarray[dtype_t, ndim=2] image,
                 cnp.ndarray[dtype_t, ndim=2] selem,
                 cnp.ndarray[dtype_t, ndim=2] mask,
                 cnp.ndarray[dtype_t, ndim=2] out,
                 char shift_x, char shift_y, float p0, float p1,
                 Py_ssize_t s0, Py_ssize_t s1) except *
