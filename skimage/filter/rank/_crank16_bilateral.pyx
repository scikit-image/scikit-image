#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

# import main loop
from skimage.filter.rank._core16 cimport _core16

# -----------------------------------------------------------------
# kernels uint16 take extra parameter for defining the bitdepth
# -----------------------------------------------------------------


cdef inline np.uint16_t kernel_mean(
    Py_ssize_t * histo, float pop, np.uint16_t g, Py_ssize_t bitdepth,
        Py_ssize_t maxbin, Py_ssize_t midbin, float p0, float p1, Py_ssize_t s0, Py_ssize_t s1):

    cdef int i, bilat_pop = 0
    cdef float mean = 0.

    if pop:
        for i in range(maxbin):
            if (g > (i - s0)) and (g < (i + s1)):
                bilat_pop += histo[i]
                mean += histo[i] * i
        if bilat_pop:
            return < np.uint16_t > (mean / bilat_pop)
        else:
            return < np.uint16_t > (0)
    else:
        return < np.uint16_t > (0)


cdef inline np.uint16_t kernel_pop(
    Py_ssize_t * histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin,
        Py_ssize_t midbin, float p0, float p1, Py_ssize_t s0, Py_ssize_t s1):

    cdef int i, bilat_pop = 0

    if pop:
        for i in range(maxbin):
            if (g > (i - s0)) and (g < (i + s1)):
                bilat_pop += histo[i]
        return < np.uint16_t > (bilat_pop)
    else:
        return < np.uint16_t > (0)


# -----------------------------------------------------------------
# python wrappers
# -----------------------------------------------------------------
def mean(np.ndarray[np.uint16_t, ndim=2] image,
         np.ndarray[np.uint8_t, ndim=2] selem,
         np.ndarray[np.uint8_t, ndim=2] mask=None,
         np.ndarray[np.uint16_t, ndim=2] out=None,
         char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
    """average gray level (clipped on uint8)
    """
    _core16(kernel_mean, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0., 0., s0, s1)


def pop(np.ndarray[np.uint16_t, ndim=2] image,
        np.ndarray[np.uint8_t, ndim=2] selem,
        np.ndarray[np.uint8_t, ndim=2] mask=None,
        np.ndarray[np.uint16_t, ndim=2] out=None,
        char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
    """returns the number of actual pixels of the structuring element inside the mask
    """
    _core16(kernel_pop, image, selem, mask, out, shift_x, shift_y,
            bitdepth, .0, .0, s0, s1)
