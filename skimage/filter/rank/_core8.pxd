cimport numpy as np


cdef np.uint8_t uint8_max(np.uint8_t a, np.uint8_t b)
cdef np.uint8_t uint8_min(np.uint8_t a, np.uint8_t b)


cdef np.uint8_t is_in_mask(Py_ssize_t rows, Py_ssize_t cols,
                           Py_ssize_t r, Py_ssize_t c,
                           np.uint8_t * mask)


# 8 bit core kernel receives extra information about data inferior and superior
# percentiles
cdef void _core8(np.uint8_t kernel(Py_ssize_t *, Py_ssize_t, np.uint8_t, float,
                                   float, Py_ssize_t, Py_ssize_t),
                 np.ndarray[np.uint8_t, ndim=2] image,
                 np.ndarray[np.uint8_t, ndim=2] selem,
                 np.ndarray[np.uint8_t, ndim=2] mask,
                 np.ndarray[np.uint8_t, ndim=2] out,
                 char shift_x, char shift_y, float p0, float p1,
                 Py_ssize_t s0, Py_ssize_t s1)
