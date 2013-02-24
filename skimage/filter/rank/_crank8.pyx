#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport log2
from skimage.filter.rank._core8 cimport _core8


# -----------------------------------------------------------------
# kernels uint8
# -----------------------------------------------------------------


ctypedef cnp.uint8_t dtype_t


cdef inline dtype_t kernel_autolevel(Py_ssize_t * histo, float pop,
                                     dtype_t g, float p0, float p1,
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
            return <dtype_t>(255. * (g - imin) / delta)
        else:
            return <dtype_t>(imax - imin)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_bottomhat(Py_ssize_t * histo, float pop,
                                     dtype_t g, float p0, float p1,
                                     Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(256):
            if histo[i]:
                break

        return <dtype_t>(g - i)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_equalize(Py_ssize_t * histo, float pop,
                                    dtype_t g, float p0, float p1,
                                    Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef float sum = 0.

    if pop:
        for i in range(256):
            sum += histo[i]
            if i >= g:
                break

        return <dtype_t>((255 * sum) / pop)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_gradient(Py_ssize_t * histo, float pop,
                                    dtype_t g, float p0, float p1,
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
        return <dtype_t>(imax - imin)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_maximum(Py_ssize_t * histo, float pop,
                                   dtype_t g, float p0, float p1,
                                   Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(255, -1, -1):
            if histo[i]:
                return <dtype_t>(i)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_mean(Py_ssize_t * histo, float pop,
                                dtype_t g, float p0, float p1,
                                Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(256):
            mean += histo[i] * i
        return <dtype_t>(mean / pop)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_meansubstraction(Py_ssize_t * histo, float pop,
                                            dtype_t g, float p0, float p1,
                                            Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(256):
            mean += histo[i] * i
        return <dtype_t>((g - mean / pop) / 2. + 127)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_median(Py_ssize_t * histo, float pop,
                                  dtype_t g, float p0, float p1,
                                  Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef float sum = pop / 2.0

    if pop:
        for i in range(256):
            if histo[i]:
                sum -= histo[i]
                if sum < 0:
                    return <dtype_t>(i)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_minimum(Py_ssize_t * histo, float pop,
                                   dtype_t g, float p0, float p1,
                                   Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(256):
            if histo[i]:
                return <dtype_t>(i)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_modal(Py_ssize_t * histo, float pop,
                                 dtype_t g, float p0, float p1,
                                 Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t hmax = 0, imax = 0

    if pop:
        for i in range(256):
            if histo[i] > hmax:
                hmax = histo[i]
                imax = i
        return <dtype_t>(imax)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_morph_contr_enh(Py_ssize_t * histo, float pop,
                                           dtype_t g, float p0, float p1,
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
            return <dtype_t>(imax)
        else:
            return <dtype_t>(imin)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_pop(Py_ssize_t * histo, float pop,
                               dtype_t g, float p0, float p1,
                               Py_ssize_t s0, Py_ssize_t s1):

    return <dtype_t>(pop)


cdef inline dtype_t kernel_threshold(Py_ssize_t * histo, float pop,
                                     dtype_t g, float p0, float p1,
                                     Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(256):
            mean += histo[i] * i
        return <dtype_t>(g > (mean / pop))
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_tophat(Py_ssize_t * histo, float pop,
                                  dtype_t g, float p0, float p1,
                                  Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(255, -1, -1):
            if histo[i]:
                break

        return <dtype_t>(i - g)
    else:
        return <dtype_t>(0)

cdef inline dtype_t kernel_noise_filter(Py_ssize_t * histo, float pop,
                                        dtype_t g, float p0, float p1,
                                        Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t min_i

    # early stop if at least one pixel of the neighborhood has the same g
    if histo[g] > 0:
        return <dtype_t>0

    for i in range(g, -1, -1):
        if histo[i]:
            break
    min_i = g - i
    for i in range(g, 256):
        if histo[i]:
            break
    if i - g < min_i:
        return <dtype_t>(i - g)
    else:
        return <dtype_t>min_i


cdef inline dtype_t kernel_entropy(Py_ssize_t * histo, float pop,
                                   dtype_t g, float p0, float p1,
                                   Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef float e, p

    if pop:
        e = 0.

        for i in range(256):
            p = histo[i] / pop
            if p > 0:
                e -= p * log2(p)

        return <dtype_t>e * 10
    else:
        return <dtype_t>(0)

cdef inline dtype_t kernel_otsu(Py_ssize_t * histo, float pop, dtype_t g,
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
        return <dtype_t>(0)

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

    return <dtype_t>max_i


# -----------------------------------------------------------------
# python wrappers
# used only internally
# -----------------------------------------------------------------


def autolevel(cnp.ndarray[dtype_t, ndim=2] image,
              cnp.ndarray[dtype_t, ndim=2] selem,
              cnp.ndarray[dtype_t, ndim=2] mask=None,
              cnp.ndarray[dtype_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0):
    _core8(kernel_autolevel, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def bottomhat(cnp.ndarray[dtype_t, ndim=2] image,
              cnp.ndarray[dtype_t, ndim=2] selem,
              cnp.ndarray[dtype_t, ndim=2] mask=None,
              cnp.ndarray[dtype_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0):
    _core8(kernel_bottomhat, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def equalize(cnp.ndarray[dtype_t, ndim=2] image,
             cnp.ndarray[dtype_t, ndim=2] selem,
             cnp.ndarray[dtype_t, ndim=2] mask=None,
             cnp.ndarray[dtype_t, ndim=2] out=None,
             char shift_x=0, char shift_y=0):
    _core8(kernel_equalize, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def gradient(cnp.ndarray[dtype_t, ndim=2] image,
             cnp.ndarray[dtype_t, ndim=2] selem,
             cnp.ndarray[dtype_t, ndim=2] mask=None,
             cnp.ndarray[dtype_t, ndim=2] out=None,
             char shift_x=0, char shift_y=0):
    _core8(kernel_gradient, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def maximum(cnp.ndarray[dtype_t, ndim=2] image,
            cnp.ndarray[dtype_t, ndim=2] selem,
            cnp.ndarray[dtype_t, ndim=2] mask=None,
            cnp.ndarray[dtype_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    _core8(kernel_maximum, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def mean(cnp.ndarray[dtype_t, ndim=2] image,
         cnp.ndarray[dtype_t, ndim=2] selem,
         cnp.ndarray[dtype_t, ndim=2] mask=None,
         cnp.ndarray[dtype_t, ndim=2] out=None,
         char shift_x=0, char shift_y=0):
    _core8(kernel_mean, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def meansubstraction(cnp.ndarray[dtype_t, ndim=2] image,
                     cnp.ndarray[dtype_t, ndim=2] selem,
                     cnp.ndarray[dtype_t, ndim=2] mask=None,
                     cnp.ndarray[dtype_t, ndim=2] out=None,
                     char shift_x=0, char shift_y=0):
    _core8(kernel_meansubstraction, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def median(cnp.ndarray[dtype_t, ndim=2] image,
           cnp.ndarray[dtype_t, ndim=2] selem,
           cnp.ndarray[dtype_t, ndim=2] mask=None,
           cnp.ndarray[dtype_t, ndim=2] out=None,
           char shift_x=0, char shift_y=0):
    _core8(kernel_median, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def minimum(cnp.ndarray[dtype_t, ndim=2] image,
            cnp.ndarray[dtype_t, ndim=2] selem,
            cnp.ndarray[dtype_t, ndim=2] mask=None,
            cnp.ndarray[dtype_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    _core8(kernel_minimum, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def morph_contr_enh(cnp.ndarray[dtype_t, ndim=2] image,
                    cnp.ndarray[dtype_t, ndim=2] selem,
                    cnp.ndarray[dtype_t, ndim=2] mask=None,
                    cnp.ndarray[dtype_t, ndim=2] out=None,
                    char shift_x=0, char shift_y=0):
    _core8(kernel_morph_contr_enh, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def modal(cnp.ndarray[dtype_t, ndim=2] image,
          cnp.ndarray[dtype_t, ndim=2] selem,
          cnp.ndarray[dtype_t, ndim=2] mask=None,
          cnp.ndarray[dtype_t, ndim=2] out=None,
          char shift_x=0, char shift_y=0):
    _core8(kernel_modal, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def pop(cnp.ndarray[dtype_t, ndim=2] image,
        cnp.ndarray[dtype_t, ndim=2] selem,
        cnp.ndarray[dtype_t, ndim=2] mask=None,
        cnp.ndarray[dtype_t, ndim=2] out=None,
        char shift_x=0, char shift_y=0):
    _core8(kernel_pop, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def threshold(cnp.ndarray[dtype_t, ndim=2] image,
              cnp.ndarray[dtype_t, ndim=2] selem,
              cnp.ndarray[dtype_t, ndim=2] mask=None,
              cnp.ndarray[dtype_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0):
    _core8(kernel_threshold, image, selem, mask, out, shift_x, shift_y, 0, 0,
           <Py_ssize_t>0, <Py_ssize_t>0)


def tophat(cnp.ndarray[dtype_t, ndim=2] image,
           cnp.ndarray[dtype_t, ndim=2] selem,
           cnp.ndarray[dtype_t, ndim=2] mask=None,
           cnp.ndarray[dtype_t, ndim=2] out=None,
           char shift_x=0, char shift_y=0):
    _core8(kernel_tophat, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def noise_filter(cnp.ndarray[dtype_t, ndim=2] image,
                 cnp.ndarray[dtype_t, ndim=2] selem,
                 cnp.ndarray[dtype_t, ndim=2] mask=None,
                 cnp.ndarray[dtype_t, ndim=2] out=None,
                 char shift_x=0, char shift_y=0):
    _core8(kernel_noise_filter, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def entropy(cnp.ndarray[dtype_t, ndim=2] image,
            cnp.ndarray[dtype_t, ndim=2] selem,
            cnp.ndarray[dtype_t, ndim=2] mask=None,
            cnp.ndarray[dtype_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    _core8(kernel_entropy, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)


def otsu(cnp.ndarray[dtype_t, ndim=2] image,
         cnp.ndarray[dtype_t, ndim=2] selem,
         cnp.ndarray[dtype_t, ndim=2] mask=None,
         cnp.ndarray[dtype_t, ndim=2] out=None,
         char shift_x=0, char shift_y=0):
    _core8(kernel_otsu, image, selem, mask, out, shift_x, shift_y,
           0, 0, <Py_ssize_t>0, <Py_ssize_t>0)
