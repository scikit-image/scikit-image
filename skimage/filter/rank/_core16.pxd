cimport numpy as cnp


ctypedef cnp.uint16_t dtype_t


cdef int int_max(int a, int b)
cdef int int_min(int a, int b)


# 16-bit core kernel receives extra information about data bitdepth
cdef void _core16(dtype_t kernel(Py_ssize_t *, float, dtype_t,
                                 Py_ssize_t, Py_ssize_t, Py_ssize_t, float,
                                 float, Py_ssize_t, Py_ssize_t),
                  cnp.ndarray[dtype_t, ndim=2] image,
                  cnp.ndarray[cnp.uint8_t, ndim=2] selem,
                  cnp.ndarray[cnp.uint8_t, ndim=2] mask,
                  cnp.ndarray[dtype_t, ndim=2] out,
                  char shift_x, char shift_y, Py_ssize_t bitdepth,
                  float p0, float p1, Py_ssize_t s0, Py_ssize_t s1) except *
