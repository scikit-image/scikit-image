#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from .core16_cy cimport dtype_t, _core16


# -----------------------------------------------------------------
# kernels uint16 take extra parameter for defining the bitdepth
# -----------------------------------------------------------------


cdef inline dtype_t kernel_mean(Py_ssize_t* histo, float pop,
                                dtype_t g, Py_ssize_t bitdepth,
                                Py_ssize_t maxbin, Py_ssize_t midbin,
                                float p0, float p1,
                                Py_ssize_t s0, Py_ssize_t s1):

    cdef int i, bilat_pop = 0
    cdef float mean = 0.

    if pop:
        for i in range(maxbin):
            if (g > (i - s0)) and (g < (i + s1)):
                bilat_pop += histo[i]
                mean += histo[i] * i
        if bilat_pop:
            return <dtype_t>(mean / bilat_pop)
        else:
            return <dtype_t>(0)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_pop(Py_ssize_t* histo, float pop,
                               dtype_t g, Py_ssize_t bitdepth,
                               Py_ssize_t maxbin, Py_ssize_t midbin,
                               float p0, float p1,
                               Py_ssize_t s0, Py_ssize_t s1):

    cdef int i, bilat_pop = 0

    if pop:
        for i in range(maxbin):
            if (g > (i - s0)) and (g < (i + s1)):
                bilat_pop += histo[i]
        return <dtype_t>(bilat_pop)
    else:
        return <dtype_t>(0)


# -----------------------------------------------------------------
# python wrappers
# -----------------------------------------------------------------


def mean(dtype_t[:, ::1] image,
         char[:, ::1] selem,
         char[:, ::1] mask=None,
         dtype_t[:, ::1] out=None,
         char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
    """average greylevel (clipped on uint8)
    """
    _core16(kernel_mean, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0., 0., s0, s1)


def pop(dtype_t[:, ::1] image,
        char[:, ::1] selem,
        char[:, ::1] mask=None,
        dtype_t[:, ::1] out=None,
        char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
    """returns the number of actual pixels of the structuring element inside
    the mask
    """
    _core16(kernel_pop, image, selem, mask, out, shift_x, shift_y,
            bitdepth, .0, .0, s0, s1)
