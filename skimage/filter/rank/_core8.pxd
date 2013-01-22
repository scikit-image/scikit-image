cimport numpy as np


cdef np.uint8_t uint8_max(np.uint8_t a, np.uint8_t b)
cdef np.uint8_t uint8_min(np.uint8_t a, np.uint8_t b)


cdef np.uint8_t is_in_mask(ssize_t rows, ssize_t cols,
                           ssize_t r, ssize_t c,
                           np.uint8_t * mask)


# 8-bit core kernel receives extra information about data inferior and superior
# percentiles
cdef void _core8(np.uint8_t kernel(ssize_t *, float, np.uint8_t, float,
                                   float, ssize_t, ssize_t),
                 np.ndarray[np.uint8_t, ndim=2] image,
                 np.ndarray[np.uint8_t, ndim=2] selem,
                 np.ndarray[np.uint8_t, ndim=2] mask,
                 np.ndarray[np.uint8_t, ndim=2] out,
                 char shift_x, char shift_y, float p0, float p1,
                 ssize_t s0, ssize_t s1) except *
