cimport numpy as np

# generic cdef functions
cdef inline np.uint8_t uint8_max(np.uint8_t a, np.uint8_t b)
cdef inline np.uint8_t uint8_min(np.uint8_t a, np.uint8_t b)

#---------------------------------------------------------------------------
# 8 bit core kernel receives extra information about data inferior and superior percentiles
#---------------------------------------------------------------------------

cdef inline _core8(np.uint8_t kernel(Py_ssize_t*, float, np.uint8_t, float, float, Py_ssize_t, Py_ssize_t),
np.ndarray[np.uint8_t, ndim=2] image,
np.ndarray[np.uint8_t, ndim=2] selem,
np.ndarray[np.uint8_t, ndim=2] mask,
np.ndarray[np.uint8_t, ndim=2] out,
char shift_x, char shift_y, float p0, float p1, Py_ssize_t s0, Py_ssize_t s1)

