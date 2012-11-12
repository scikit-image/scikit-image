#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport log2
from skimage.filter.rank._core16 cimport _core16


# -----------------------------------------------------------------
# kernels uint16 take extra parameter for defining the bitdepth
# -----------------------------------------------------------------


cdef inline np.uint16_t kernel_autolevel(Py_ssize_t * histo, float pop,
                                         np.uint16_t g, Py_ssize_t bitdepth,
                                         Py_ssize_t maxbin, Py_ssize_t midbin,
                                         float p0, float p1,
                                         Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i, imin, imax, delta

    if pop:
        for i in range(maxbin - 1, -1, -1):
            if histo[i]:
                imax = i
                break
        for i in range(maxbin):
            if histo[i]:
                imin = i
                break
    delta = imax - imin
    if delta > 0:
        return <np.uint16_t>(1. * (maxbin - 1) * (g - imin) / delta)
    else:
        return <np.uint16_t>(imax - imin)


cdef inline np.uint16_t kernel_bottomhat(Py_ssize_t * histo, float pop,
                                         np.uint16_t g, Py_ssize_t bitdepth,
                                         Py_ssize_t maxbin, Py_ssize_t midbin,
                                         float p0, float p1,
                                         Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i

    if pop:
        for i in range(maxbin):
            if histo[i]:
                break

        return <np.uint16_t>(g - i)
    else:
        return <np.uint16_t>(0)

cdef inline np.uint16_t kernel_equalize(Py_ssize_t * histo, float pop,
                                        np.uint16_t g, Py_ssize_t bitdepth,
                                        Py_ssize_t maxbin, Py_ssize_t midbin,
                                        float p0, float p1,
                                        Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef float sum = 0.

    if pop:
        for i in range(maxbin):
            sum += histo[i]
            if i >= g:
                break

        return <np.uint16_t>(((maxbin - 1) * sum) / pop)
    else:
        return <np.uint16_t>(0)


cdef inline np.uint16_t kernel_gradient(Py_ssize_t * histo, float pop,
                                        np.uint16_t g, Py_ssize_t bitdepth,
                                        Py_ssize_t maxbin, Py_ssize_t midbin,
                                        float p0, float p1,
                                        Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i, imin, imax

    if pop:
        for i in range(maxbin - 1, -1, -1):
            if histo[i]:
                imax = i
                break
        for i in range(maxbin):
            if histo[i]:
                imin = i
                break
        return <np.uint16_t>(imax - imin)
    else:
        return <np.uint16_t>(0)


cdef inline np.uint16_t kernel_maximum(Py_ssize_t * histo, float pop,
                                       np.uint16_t g, Py_ssize_t bitdepth,
                                       Py_ssize_t maxbin, Py_ssize_t midbin,
                                       float p0, float p1,
                                       Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i

    if pop:
        for i in range(maxbin - 1, -1, -1):
            if histo[i]:
                return <np.uint16_t>(i)

    return <np.uint16_t>(0)


cdef inline np.uint16_t kernel_mean(Py_ssize_t * histo, float pop,
                                    np.uint16_t g, Py_ssize_t bitdepth,
                                    Py_ssize_t maxbin, Py_ssize_t midbin,
                                    float p0, float p1,
                                    Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(maxbin):
            mean += histo[i] * i
        return <np.uint16_t>(mean / pop)
    else:
        return <np.uint16_t>(0)


cdef inline np.uint16_t kernel_meansubstraction(Py_ssize_t * histo,
                                                float pop,
                                                np.uint16_t g,
                                                Py_ssize_t bitdepth,
                                                Py_ssize_t maxbin,
                                                Py_ssize_t midbin,
                                                float p0, float p1,
                                                Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(maxbin):
            mean += histo[i] * i
        return <np.uint16_t>((g - mean / pop) / 2. + (midbin - 1))
    else:
        return <np.uint16_t>(0)


cdef inline np.uint16_t kernel_median(Py_ssize_t * histo, float pop,
                                      np.uint16_t g, Py_ssize_t bitdepth,
                                      Py_ssize_t maxbin, Py_ssize_t midbin,
                                      float p0, float p1,
                                      Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef float sum = pop / 2.0

    if pop:
        for i in range(maxbin):
            if histo[i]:
                sum -= histo[i]
                if sum < 0:
                    return <np.uint16_t>(i)
    else:
        return <np.uint16_t>(0)


cdef inline np.uint16_t kernel_minimum(Py_ssize_t * histo, float pop,
                                       np.uint16_t g, Py_ssize_t bitdepth,
                                       Py_ssize_t maxbin, Py_ssize_t midbin,
                                       float p0, float p1,
                                       Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i

    if pop:
        for i in range(maxbin):
            if histo[i]:
                return <np.uint16_t>(i)
    else:
        return <np.uint16_t>(0)


cdef inline np.uint16_t kernel_modal(Py_ssize_t * histo, float pop,
                                     np.uint16_t g, Py_ssize_t bitdepth,
                                     Py_ssize_t maxbin, Py_ssize_t midbin,
                                     float p0, float p1,
                                     Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t hmax = 0, imax = 0

    if pop:
        for i in range(maxbin):
            if histo[i] > hmax:
                hmax = histo[i]
                imax = i
        return <np.uint16_t>(imax)
    else:
        return <np.uint16_t>(0)


cdef inline np.uint16_t kernel_morph_contr_enh(Py_ssize_t * histo,
                                               float pop,
                                               np.uint16_t g,
                                               Py_ssize_t bitdepth,
                                               Py_ssize_t maxbin,
                                               Py_ssize_t midbin,
                                               float p0, float p1,
                                               Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i, imin, imax

    if pop:
        for i in range(maxbin - 1, -1, -1):
            if histo[i]:
                imax = i
                break
        for i in range(maxbin):
            if histo[i]:
                imin = i
                break
        if imax - g < g - imin:
            return <np.uint16_t>(imax)
        else:
            return <np.uint16_t>(imin)
    else:
        return <np.uint16_t>(0)


cdef inline np.uint16_t kernel_pop(Py_ssize_t * histo, float pop,
                                   np.uint16_t g, Py_ssize_t bitdepth,
                                   Py_ssize_t maxbin, Py_ssize_t midbin,
                                   float p0, float p1,
                                   Py_ssize_t s0, Py_ssize_t s1):
    return <np.uint16_t>(pop)


cdef inline np.uint16_t kernel_threshold(Py_ssize_t * histo, float pop,
                                         np.uint16_t g, Py_ssize_t bitdepth,
                                         Py_ssize_t maxbin, Py_ssize_t midbin,
                                         float p0, float p1,
                                         Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(maxbin):
            mean += histo[i] * i
        return <np.uint16_t>(g > (mean / pop))
    else:
        return <np.uint16_t>(0)


cdef inline np.uint16_t kernel_tophat(Py_ssize_t * histo, float pop,
                                      np.uint16_t g, Py_ssize_t bitdepth,
                                      Py_ssize_t maxbin, Py_ssize_t midbin,
                                      float p0, float p1,
                                      Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i

    if pop:
        for i in range(maxbin - 1, -1, -1):
            if histo[i]:
                break

        return <np.uint16_t>(i - g)
    else:
        return <np.uint16_t>(0)

cdef inline np.uint16_t kernel_entropy(Py_ssize_t * histo, float pop,
                                       np.uint16_t g, Py_ssize_t bitdepth,
                                       Py_ssize_t maxbin, Py_ssize_t midbin,
                                       float p0, float p1,
                                       Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef float e,p

    if pop:
        e = 0.

        for i in range(maxbin):
            p = histo[i]/pop
            if p>0:
                e -= p*log2(p)

        return <np.uint16_t>e*1000
    else:
        return <np.uint16_t>(0)

# -----------------------------------------------------------------
# python wrappers
# -----------------------------------------------------------------


def autolevel(np.ndarray[np.uint16_t, ndim=2] image,
              np.ndarray[np.uint8_t, ndim=2] selem,
              np.ndarray[np.uint8_t, ndim=2] mask=None,
              np.ndarray[np.uint16_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_autolevel, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def bottomhat(np.ndarray[np.uint16_t, ndim=2] image,
              np.ndarray[np.uint8_t, ndim=2] selem,
              np.ndarray[np.uint8_t, ndim=2] mask=None,
              np.ndarray[np.uint16_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_bottomhat, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def equalize(np.ndarray[np.uint16_t, ndim=2] image,
             np.ndarray[np.uint8_t, ndim=2] selem,
             np.ndarray[np.uint8_t, ndim=2] mask=None,
             np.ndarray[np.uint16_t, ndim=2] out=None,
             char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_equalize, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def gradient(np.ndarray[np.uint16_t, ndim=2] image,
             np.ndarray[np.uint8_t, ndim=2] selem,
             np.ndarray[np.uint8_t, ndim=2] mask=None,
             np.ndarray[np.uint16_t, ndim=2] out=None,
             char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_gradient, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def maximum(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_maximum, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def mean(np.ndarray[np.uint16_t, ndim=2] image,
         np.ndarray[np.uint8_t, ndim=2] selem,
         np.ndarray[np.uint8_t, ndim=2] mask=None,
         np.ndarray[np.uint16_t, ndim=2] out=None,
         char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_mean, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def meansubstraction(np.ndarray[np.uint16_t, ndim=2] image,
                     np.ndarray[np.uint8_t, ndim=2] selem,
                     np.ndarray[np.uint8_t, ndim=2] mask=None,
                     np.ndarray[np.uint16_t, ndim=2] out=None,
                     char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_meansubstraction, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def median(np.ndarray[np.uint16_t, ndim=2] image,
           np.ndarray[np.uint8_t, ndim=2] selem,
           np.ndarray[np.uint8_t, ndim=2] mask=None,
           np.ndarray[np.uint16_t, ndim=2] out=None,
           char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_median, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def minimum(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_minimum, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def morph_contr_enh(np.ndarray[np.uint16_t, ndim=2] image,
                    np.ndarray[np.uint8_t, ndim=2] selem,
                    np.ndarray[np.uint8_t, ndim=2] mask=None,
                    np.ndarray[np.uint16_t, ndim=2] out=None,
                    char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_morph_contr_enh, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def modal(np.ndarray[np.uint16_t, ndim=2] image,
          np.ndarray[np.uint8_t, ndim=2] selem,
          np.ndarray[np.uint8_t, ndim=2] mask=None,
          np.ndarray[np.uint16_t, ndim=2] out=None,
          char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_modal, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def pop(np.ndarray[np.uint16_t, ndim=2] image,
        np.ndarray[np.uint8_t, ndim=2] selem,
        np.ndarray[np.uint8_t, ndim=2] mask=None,
        np.ndarray[np.uint16_t, ndim=2] out=None,
        char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_pop, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def threshold(np.ndarray[np.uint16_t, ndim=2] image,
              np.ndarray[np.uint8_t, ndim=2] selem,
              np.ndarray[np.uint8_t, ndim=2] mask=None,
              np.ndarray[np.uint16_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_threshold, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def tophat(np.ndarray[np.uint16_t, ndim=2] image,
           np.ndarray[np.uint8_t, ndim=2] selem,
           np.ndarray[np.uint8_t, ndim=2] mask=None,
           np.ndarray[np.uint16_t, ndim=2] out=None,
           char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_tophat, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)

def entropy(np.ndarray[np.uint16_t, ndim=2] image,
           np.ndarray[np.uint8_t, ndim=2] selem,
           np.ndarray[np.uint8_t, ndim=2] mask=None,
           np.ndarray[np.uint16_t, ndim=2] out=None,
           char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_entropy, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)
