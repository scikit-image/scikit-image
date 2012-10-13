cimport numpy as np

#---------------------------------------------------------------------------
# 8 bit core kernel
#---------------------------------------------------------------------------

cdef inline _core8(np.uint8_t kernel(int*, float, np.uint8_t),
np.ndarray[np.uint8_t, ndim=2] image,
np.ndarray[np.uint8_t, ndim=2] selem,
np.ndarray[np.uint8_t, ndim=2] mask,
np.ndarray[np.uint8_t, ndim=2] out,
char shift_x, char shift_y)
