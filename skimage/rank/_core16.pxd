cimport numpy as np

#---------------------------------------------------------------------------
# 16 bit core kernel receives extra information about data bitdepth
#---------------------------------------------------------------------------

cdef inline _core16(np.uint16_t kernel(Py_ssize_t*, float, np.uint16_t, Py_ssize_t ,Py_ssize_t,Py_ssize_t ),
np.ndarray[np.uint16_t, ndim=2] image,
np.ndarray[np.uint8_t, ndim=2] selem,
np.ndarray[np.uint8_t, ndim=2] mask,
np.ndarray[np.uint16_t, ndim=2] out,
char shift_x, char shift_y,Py_ssize_t bitdepth)