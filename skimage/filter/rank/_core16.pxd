cimport numpy as np


cdef int int_max(int a, int b)
cdef int int_min(int a, int b)


# 16-bit core kernel receives extra information about data bitdepth
cdef void _core16(np.uint16_t kernel(ssize_t *, float, np.uint16_t,
                                     ssize_t, ssize_t, ssize_t, float,
                                     float, ssize_t, ssize_t),
                  np.ndarray[np.uint16_t, ndim=2] image,
                  np.ndarray[np.uint8_t, ndim=2] selem,
                  np.ndarray[np.uint8_t, ndim=2] mask,
                  np.ndarray[np.uint16_t, ndim=2] out,
                  char shift_x, char shift_y, ssize_t bitdepth,
                  float p0, float p1, ssize_t s0, ssize_t s1) except *
