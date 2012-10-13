cimport numpy as np

#---------------------------------------------------------------------------
# 16 bit core kernel receives extra information about data bitdepth and bilateral interval
#---------------------------------------------------------------------------

cdef inline _core16b(np.uint16_t kernel(int*, float, np.uint16_t, int ,int,int,int,int),
np.ndarray[np.uint16_t, ndim=2] image,
np.ndarray[np.uint8_t, ndim=2] selem,
np.ndarray[np.uint8_t, ndim=2] mask,
np.ndarray[np.uint16_t, ndim=2] out,
char shift_x, char shift_y,int bitdepth, int s0, int s1)