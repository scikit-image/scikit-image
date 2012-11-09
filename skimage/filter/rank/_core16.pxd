cimport numpy as np

#---------------------------------------------------------------------------
# 16 bit core kernel receives extra information about data bitdepth
#---------------------------------------------------------------------------

# generic cdef functions
cdef int int_max(int a, int b)
cdef int int_min(int a, int b)

cdef _core16(
    np.uint16_t kernel(Py_ssize_t * , float, np.uint16_t, Py_ssize_t, Py_ssize_t, Py_ssize_t, float, float, Py_ssize_t, Py_ssize_t),
    np.ndarray[np.uint16_t, ndim=2] image,
    np.ndarray[np.uint8_t, ndim=2] selem,
    np.ndarray[np.uint8_t, ndim=2] mask,
    np.ndarray[np.uint16_t, ndim=2] out,
    char shift_x, char shift_y, Py_ssize_t bitdepth,
    float p0, float p1, Py_ssize_t s0, Py_ssize_t s1)
