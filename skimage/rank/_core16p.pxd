cimport numpy as np

# generic cdef functions
cdef inline int int_max(int a, int b)
cdef inline int int_min(int a, int b)

#---------------------------------------------------------------------------
# 16 bit core kernel receives extra information about data inferior and superior percentiles
#---------------------------------------------------------------------------

cdef inline _core16p(np.uint16_t kernel(int*, float, np.uint16_t,int,int,int, float, float),
np.ndarray[np.uint16_t, ndim=2] image,
np.ndarray[np.uint8_t, ndim=2] selem,
np.ndarray[np.uint8_t, ndim=2] mask,
np.ndarray[np.uint16_t, ndim=2] out,
char shift_x, char shift_y,int bitdepth, float p0, float p1)