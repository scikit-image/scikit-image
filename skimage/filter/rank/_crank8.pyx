#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport log
from skimage.filter.rank._core8 cimport _core8


# -----------------------------------------------------------------
# kernels uint8
# -----------------------------------------------------------------


cdef inline np.uint8_t kernel_autolevel(Py_ssize_t * histo, float pop,
                                        np.uint8_t g, float p0, float p1,
                                        Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i, imin, imax, delta

    if pop:
        for i in range(255, -1, -1):
            if histo[i]:
                imax = i
                break
        for i in range(256):
            if histo[i]:
                imin = i
                break
        delta = imax - imin
        if delta > 0:
            return < np.uint8_t > (255. * (g - imin) / delta)
        else:
            return < np.uint8_t > (imax - imin)
    else:
        return < np.uint8_t > (0)


cdef inline np.uint8_t kernel_bottomhat(Py_ssize_t * histo, float pop,
                                        np.uint8_t g, float p0, float p1,
                                        Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(256):
            if histo[i]:
                break

        return < np.uint8_t > (g - i)
    else:
        return < np.uint8_t > (0)


cdef inline np.uint8_t kernel_equalize(Py_ssize_t * histo, float pop,
                                       np.uint8_t g, float p0, float p1,
                                       Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef float sum = 0.

    if pop:
        for i in range(256):
            sum += histo[i]
            if i >= g:
                break

        return < np.uint8_t > ((255 * sum) / pop)
    else:
        return < np.uint8_t > (0)


cdef inline np.uint8_t kernel_gradient(Py_ssize_t * histo, float pop,
                                       np.uint8_t g, float p0, float p1,
                                       Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i, imin, imax

    if pop:
        for i in range(255, -1, -1):
            if histo[i]:
                imax = i
                break
        for i in range(256):
            if histo[i]:
                imin = i
                break
        return < np.uint8_t > (imax - imin)
    else:
        return < np.uint8_t > (0)


cdef inline np.uint8_t kernel_maximum(Py_ssize_t * histo, float pop,
                                      np.uint8_t g, float p0, float p1,
                                      Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(255, -1, -1):
            if histo[i]:
                return < np.uint8_t > (i)
    else:
        return < np.uint8_t > (0)


cdef inline np.uint8_t kernel_mean(Py_ssize_t * histo, float pop,
                                   np.uint8_t g, float p0, float p1,
                                   Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(256):
            mean += histo[i] * i
        return < np.uint8_t > (mean / pop)
    else:
        return < np.uint8_t > (0)


cdef inline np.uint8_t kernel_meansubstraction(Py_ssize_t * histo, float pop,
                                               np.uint8_t g, float p0, float p1,
                                               Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(256):
            mean += histo[i] * i
        return < np.uint8_t > ((g - mean / pop) / 2. + 127)
    else:
        return < np.uint8_t > (0)


cdef inline np.uint8_t kernel_median(Py_ssize_t * histo, float pop,
                                     np.uint8_t g, float p0, float p1,
                                     Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef float sum = pop / 2.0

    if pop:
        for i in range(256):
            if histo[i]:
                sum -= histo[i]
                if sum < 0:
                    return < np.uint8_t > (i)
    else:
        return < np.uint8_t > (0)


cdef inline np.uint8_t kernel_minimum(Py_ssize_t * histo, float pop,
                                      np.uint8_t g, float p0, float p1,
                                      Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(256):
            if histo[i]:
                return < np.uint8_t > (i)
    else:
        return < np.uint8_t > (0)


cdef inline np.uint8_t kernel_modal(Py_ssize_t * histo, float pop,
                                    np.uint8_t g, float p0, float p1,
                                    Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t hmax = 0, imax = 0

    if pop:
        for i in range(256):
            if histo[i] > hmax:
                hmax = histo[i]
                imax = i
        return < np.uint8_t > (imax)
    else:
        return < np.uint8_t > (0)


cdef inline np.uint8_t kernel_morph_contr_enh(Py_ssize_t * histo, float pop,
                                              np.uint8_t g, float p0, float p1,
                                              Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i, imin, imax

    if pop:
        for i in range(255, -1, -1):
            if histo[i]:
                imax = i
                break
        for i in range(256):
            if histo[i]:
                imin = i
                break
        if imax - g < g - imin:
            return < np.uint8_t > (imax)
        else:
            return < np.uint8_t > (imin)
    else:
        return < np.uint8_t > (0)


cdef inline np.uint8_t kernel_pop(Py_ssize_t * histo, float pop,
                                  np.uint8_t g, float p0, float p1,
                                  Py_ssize_t s0, Py_ssize_t s1):

    return < np.uint8_t > (pop)


cdef inline np.uint8_t kernel_threshold(Py_ssize_t * histo, float pop,
                                        np.uint8_t g, float p0, float p1,
                                        Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(256):
            mean += histo[i] * i
        return < np.uint8_t > (g > (mean / pop))
    else:
        return < np.uint8_t > (0)


cdef inline np.uint8_t kernel_tophat(Py_ssize_t * histo, float pop,
                                     np.uint8_t g, float p0, float p1,
                                     Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(255, -1, -1):
            if histo[i]:
                break

        return < np.uint8_t > (i - g)
    else:
        return < np.uint8_t > (0)

cdef inline np.uint8_t kernel_noise_filter(Py_ssize_t * histo, float pop,
                                           np.uint8_t g, float p0, float p1,
                                           Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t min_i

    # early stop if at least one pixel of the neighborhood has the same g
    if histo[g] > 0:
        return < np.uint8_t > 0

    for i in range(g, -1, -1):
        if histo[i]:
            break
    min_i = g - i
    for i in range(g, 256):
        if histo[i]:
            break
    if i - g < min_i:
        return < np.uint8_t > (i - g)
    else:
        return < np.uint8_t > min_i


cdef inline np.uint8_t kernel_entropy(Py_ssize_t * histo, float pop,
                                      np.uint8_t g, float p0, float p1,
                                      Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef float e, p

    if pop:
        e = 0.

        for i in range(256):
            p = histo[i] / pop
            if p > 0:
                e -= p * log(p) / 0.30102999566398119521373889472449

        return < np.uint8_t > e * 10
    else:
        return < np.uint8_t > (0)

cdef inline np.uint8_t kernel_otsu(Py_ssize_t * histo, float pop, np.uint8_t g,
                                   float p0, float p1, Py_ssize_t s0,
                                   Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef Py_ssize_t max_i
    cdef float P, mu1, mu2, q1, new_q1, sigma_b, max_sigma_b
    cdef float mu = 0.

    # compute local mean
    if pop:
        for i in range(256):
            mu += histo[i] * i
        mu = (mu / pop)
    else:
        return < np.uint8_t > (0)

    # maximizing the between class variance
    max_i = 0
    q1 = histo[0] / pop
    m1 = 0.
    max_sigma_b = 0.

    for i in range(1, 256):
        P = histo[i] / pop
        new_q1 = q1 + P
        if new_q1 > 0:
            mu1 = (q1 * mu1 + i * P) / new_q1
            mu2 = (mu - new_q1 * mu1) / (1. - new_q1)
            sigma_b = new_q1 * (1. - new_q1) * (mu1 - mu2) ** 2
            if sigma_b > max_sigma_b:
                max_sigma_b = sigma_b
                max_i = i
            q1 = new_q1

    return < np.uint8_t > max_i


# -----------------------------------------------------------------
# python wrappers
# used only internally
# -----------------------------------------------------------------


def autolevel(np.ndarray[np.uint8_t, ndim=2] image,
              np.ndarray[np.uint8_t, ndim=2] selem,
              np.ndarray[np.uint8_t, ndim=2] mask=None,
              np.ndarray[np.uint8_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0):
    _core8(kernel_autolevel, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def bottomhat(np.ndarray[np.uint8_t, ndim=2] image,
              np.ndarray[np.uint8_t, ndim=2] selem,
              np.ndarray[np.uint8_t, ndim=2] mask=None,
              np.ndarray[np.uint8_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0):
    _core8(kernel_bottomhat, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def equalize(np.ndarray[np.uint8_t, ndim=2] image,
             np.ndarray[np.uint8_t, ndim=2] selem,
             np.ndarray[np.uint8_t, ndim=2] mask=None,
             np.ndarray[np.uint8_t, ndim=2] out=None,
             char shift_x=0, char shift_y=0):
    _core8(kernel_equalize, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def gradient(np.ndarray[np.uint8_t, ndim=2] image,
             np.ndarray[np.uint8_t, ndim=2] selem,
             np.ndarray[np.uint8_t, ndim=2] mask=None,
             np.ndarray[np.uint8_t, ndim=2] out=None,
             char shift_x=0, char shift_y=0):
    _core8(kernel_gradient, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def maximum(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    _core8(kernel_maximum, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def mean(np.ndarray[np.uint8_t, ndim=2] image,
         np.ndarray[np.uint8_t, ndim=2] selem,
         np.ndarray[np.uint8_t, ndim=2] mask=None,
         np.ndarray[np.uint8_t, ndim=2] out=None,
         char shift_x=0, char shift_y=0):
    _core8(kernel_mean, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def meansubstraction(np.ndarray[np.uint8_t, ndim=2] image,
                     np.ndarray[np.uint8_t, ndim=2] selem,
                     np.ndarray[np.uint8_t, ndim=2] mask=None,
                     np.ndarray[np.uint8_t, ndim=2] out=None,
                     char shift_x=0, char shift_y=0):
    _core8(kernel_meansubstraction, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def median(np.ndarray[np.uint8_t, ndim=2] image,
           np.ndarray[np.uint8_t, ndim=2] selem,
           np.ndarray[np.uint8_t, ndim=2] mask=None,
           np.ndarray[np.uint8_t, ndim=2] out=None,
           char shift_x=0, char shift_y=0):
    _core8(kernel_median, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def minimum(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    _core8(kernel_minimum, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def morph_contr_enh(np.ndarray[np.uint8_t, ndim=2] image,
                    np.ndarray[np.uint8_t, ndim=2] selem,
                    np.ndarray[np.uint8_t, ndim=2] mask=None,
                    np.ndarray[np.uint8_t, ndim=2] out=None,
                    char shift_x=0, char shift_y=0):
    _core8(kernel_morph_contr_enh, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def modal(np.ndarray[np.uint8_t, ndim=2] image,
          np.ndarray[np.uint8_t, ndim=2] selem,
          np.ndarray[np.uint8_t, ndim=2] mask=None,
          np.ndarray[np.uint8_t, ndim=2] out=None,
          char shift_x=0, char shift_y=0):
    _core8(kernel_modal, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def pop(np.ndarray[np.uint8_t, ndim=2] image,
        np.ndarray[np.uint8_t, ndim=2] selem,
        np.ndarray[np.uint8_t, ndim=2] mask=None,
        np.ndarray[np.uint8_t, ndim=2] out=None,
        char shift_x=0, char shift_y=0):
    _core8(kernel_pop, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def threshold(np.ndarray[np.uint8_t, ndim=2] image,
              np.ndarray[np.uint8_t, ndim=2] selem,
              np.ndarray[np.uint8_t, ndim=2] mask=None,
              np.ndarray[np.uint8_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0):
    _core8(kernel_threshold, image, selem, mask, out, shift_x, shift_y, 0, 0,
           <Py_ssize_t>0, <Py_ssize_t>0)


def tophat(np.ndarray[np.uint8_t, ndim=2] image,
           np.ndarray[np.uint8_t, ndim=2] selem,
           np.ndarray[np.uint8_t, ndim=2] mask=None,
           np.ndarray[np.uint8_t, ndim=2] out=None,
           char shift_x=0, char shift_y=0):
    _core8(kernel_tophat, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def noise_filter(np.ndarray[np.uint8_t, ndim=2] image,
                 np.ndarray[np.uint8_t, ndim=2] selem,
                 np.ndarray[np.uint8_t, ndim=2] mask=None,
                 np.ndarray[np.uint8_t, ndim=2] out=None,
                 char shift_x=0, char shift_y=0):
    _core8(kernel_noise_filter, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def entropy(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    _core8(kernel_entropy, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def otsu(np.ndarray[np.uint8_t, ndim=2] image,
         np.ndarray[np.uint8_t, ndim=2] selem,
         np.ndarray[np.uint8_t, ndim=2] mask=None,
         np.ndarray[np.uint8_t, ndim=2] out=None,
         char shift_x=0, char shift_y=0):
    _core8(kernel_otsu, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)
