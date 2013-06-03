#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport log
from skimage.filter.rank._core16 cimport _core16


# -----------------------------------------------------------------
# kernels uint16 take extra parameter for defining the bitdepth
# -----------------------------------------------------------------


ctypedef cnp.uint16_t dtype_t


cdef inline dtype_t kernel_autolevel(Py_ssize_t * histo, float pop,
                                     dtype_t g, Py_ssize_t bitdepth,
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
        return <dtype_t>(1. * (maxbin - 1) * (g - imin) / delta)
    else:
        return <dtype_t>(imax - imin)


cdef inline dtype_t kernel_bottomhat(Py_ssize_t * histo, float pop,
                                     dtype_t g, Py_ssize_t bitdepth,
                                     Py_ssize_t maxbin, Py_ssize_t midbin,
                                     float p0, float p1,
                                     Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i

    if pop:
        for i in range(maxbin):
            if histo[i]:
                break

        return <dtype_t>(g - i)
    else:
        return <dtype_t>(0)

cdef inline dtype_t kernel_equalize(Py_ssize_t * histo, float pop,
                                    dtype_t g, Py_ssize_t bitdepth,
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

        return <dtype_t>(((maxbin - 1) * sum) / pop)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_gradient(Py_ssize_t * histo, float pop,
                                    dtype_t g, Py_ssize_t bitdepth,
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
        return <dtype_t>(imax - imin)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_maximum(Py_ssize_t * histo, float pop,
                                   dtype_t g, Py_ssize_t bitdepth,
                                   Py_ssize_t maxbin, Py_ssize_t midbin,
                                   float p0, float p1,
                                   Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i

    if pop:
        for i in range(maxbin - 1, -1, -1):
            if histo[i]:
                return <dtype_t>(i)

    return <dtype_t>(0)


cdef inline dtype_t kernel_mean(Py_ssize_t * histo, float pop,
                                dtype_t g, Py_ssize_t bitdepth,
                                Py_ssize_t maxbin, Py_ssize_t midbin,
                                float p0, float p1,
                                Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(maxbin):
            mean += histo[i] * i
        return <dtype_t>(mean / pop)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_meansubstraction(Py_ssize_t * histo,
                                            float pop,
                                            dtype_t g,
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
        return <dtype_t>((g - mean / pop) / 2. + (midbin - 1))
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_median(Py_ssize_t * histo, float pop,
                                  dtype_t g, Py_ssize_t bitdepth,
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
                    return <dtype_t>(i)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_minimum(Py_ssize_t * histo, float pop,
                                   dtype_t g, Py_ssize_t bitdepth,
                                   Py_ssize_t maxbin, Py_ssize_t midbin,
                                   float p0, float p1,
                                   Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i

    if pop:
        for i in range(maxbin):
            if histo[i]:
                return <dtype_t>(i)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_modal(Py_ssize_t * histo, float pop,
                                 dtype_t g, Py_ssize_t bitdepth,
                                 Py_ssize_t maxbin, Py_ssize_t midbin,
                                 float p0, float p1,
                                 Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t hmax = 0, imax = 0

    if pop:
        for i in range(maxbin):
            if histo[i] > hmax:
                hmax = histo[i]
                imax = i
        return <dtype_t>(imax)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_morph_contr_enh(Py_ssize_t * histo,
                                           float pop,
                                           dtype_t g,
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
            return <dtype_t>(imax)
        else:
            return <dtype_t>(imin)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_pop(Py_ssize_t * histo, float pop,
                               dtype_t g, Py_ssize_t bitdepth,
                               Py_ssize_t maxbin, Py_ssize_t midbin,
                               float p0, float p1,
                               Py_ssize_t s0, Py_ssize_t s1):
    return <dtype_t>(pop)


cdef inline dtype_t kernel_threshold(Py_ssize_t * histo, float pop,
                                     dtype_t g, Py_ssize_t bitdepth,
                                     Py_ssize_t maxbin, Py_ssize_t midbin,
                                     float p0, float p1,
                                     Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(maxbin):
            mean += histo[i] * i
        return <dtype_t>(g > (mean / pop))
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_tophat(Py_ssize_t * histo, float pop,
                                  dtype_t g, Py_ssize_t bitdepth,
                                  Py_ssize_t maxbin, Py_ssize_t midbin,
                                  float p0, float p1,
                                  Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i

    if pop:
        for i in range(maxbin - 1, -1, -1):
            if histo[i]:
                break

        return <dtype_t>(i - g)
    else:
        return <dtype_t>(0)

cdef inline dtype_t kernel_entropy(Py_ssize_t * histo, float pop,
                                   dtype_t g, Py_ssize_t bitdepth,
                                   Py_ssize_t maxbin, Py_ssize_t midbin,
                                   float p0, float p1,
                                   Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef float e, p

    if pop:
        e = 0.

        for i in range(maxbin):
            p = histo[i] / pop
            if p > 0:
                e -= p * log(p) / 0.6931471805599453

        return <dtype_t>e * 1000
    else:
        return <dtype_t>(0)

# -----------------------------------------------------------------
# python wrappers
# -----------------------------------------------------------------


def autolevel(cnp.ndarray[dtype_t, ndim=2] image,
              cnp.ndarray[cnp.uint8_t, ndim=2] selem,
              cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
              cnp.ndarray[dtype_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_autolevel, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def bottomhat(cnp.ndarray[dtype_t, ndim=2] image,
              cnp.ndarray[cnp.uint8_t, ndim=2] selem,
              cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
              cnp.ndarray[dtype_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_bottomhat, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def equalize(cnp.ndarray[dtype_t, ndim=2] image,
             cnp.ndarray[cnp.uint8_t, ndim=2] selem,
             cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
             cnp.ndarray[dtype_t, ndim=2] out=None,
             char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_equalize, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def gradient(cnp.ndarray[dtype_t, ndim=2] image,
             cnp.ndarray[cnp.uint8_t, ndim=2] selem,
             cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
             cnp.ndarray[dtype_t, ndim=2] out=None,
             char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_gradient, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def maximum(cnp.ndarray[dtype_t, ndim=2] image,
            cnp.ndarray[cnp.uint8_t, ndim=2] selem,
            cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
            cnp.ndarray[dtype_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_maximum, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def mean(cnp.ndarray[dtype_t, ndim=2] image,
         cnp.ndarray[cnp.uint8_t, ndim=2] selem,
         cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
         cnp.ndarray[dtype_t, ndim=2] out=None,
         char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_mean, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def meansubstraction(cnp.ndarray[dtype_t, ndim=2] image,
                     cnp.ndarray[cnp.uint8_t, ndim=2] selem,
                     cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
                     cnp.ndarray[dtype_t, ndim=2] out=None,
                     char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_meansubstraction, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def median(cnp.ndarray[dtype_t, ndim=2] image,
           cnp.ndarray[cnp.uint8_t, ndim=2] selem,
           cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
           cnp.ndarray[dtype_t, ndim=2] out=None,
           char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_median, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def minimum(cnp.ndarray[dtype_t, ndim=2] image,
            cnp.ndarray[cnp.uint8_t, ndim=2] selem,
            cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
            cnp.ndarray[dtype_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_minimum, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def morph_contr_enh(cnp.ndarray[dtype_t, ndim=2] image,
                    cnp.ndarray[cnp.uint8_t, ndim=2] selem,
                    cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
                    cnp.ndarray[dtype_t, ndim=2] out=None,
                    char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_morph_contr_enh, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def modal(cnp.ndarray[dtype_t, ndim=2] image,
          cnp.ndarray[cnp.uint8_t, ndim=2] selem,
          cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
          cnp.ndarray[dtype_t, ndim=2] out=None,
          char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_modal, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def pop(cnp.ndarray[dtype_t, ndim=2] image,
        cnp.ndarray[cnp.uint8_t, ndim=2] selem,
        cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
        cnp.ndarray[dtype_t, ndim=2] out=None,
        char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_pop, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def threshold(cnp.ndarray[dtype_t, ndim=2] image,
              cnp.ndarray[cnp.uint8_t, ndim=2] selem,
              cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
              cnp.ndarray[dtype_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_threshold, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def tophat(cnp.ndarray[dtype_t, ndim=2] image,
           cnp.ndarray[cnp.uint8_t, ndim=2] selem,
           cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
           cnp.ndarray[dtype_t, ndim=2] out=None,
           char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_tophat, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def entropy(cnp.ndarray[dtype_t, ndim=2] image,
            cnp.ndarray[cnp.uint8_t, ndim=2] selem,
            cnp.ndarray[cnp.uint8_t, ndim=2] mask=None,
            cnp.ndarray[dtype_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    _core16(kernel_entropy, image, selem, mask, out, shift_x, shift_y,
            bitdepth, 0, 0, <Py_ssize_t>0, <Py_ssize_t>0)
