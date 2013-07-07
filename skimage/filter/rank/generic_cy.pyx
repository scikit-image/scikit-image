#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport log

from .core_cy cimport uint8_t, uint16_t, dtype_t, _core


cdef inline dtype_t _kernel_autolevel(Py_ssize_t* histo, float pop, dtype_t g,
                                      Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                      float p0, float p1,
                                      Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i, imin, imax, delta

    if pop:
        for i in range(max_bin - 1, -1, -1):
            if histo[i]:
                imax = i
                break
        for i in range(max_bin):
            if histo[i]:
                imin = i
                break
        delta = imax - imin
        if delta > 0:
            return <dtype_t>(<float>(max_bin - 1) * (g - imin) / delta)
        else:
            return <dtype_t>(imax - imin)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_bottomhat(Py_ssize_t* histo, float pop, dtype_t g,
                                      Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                      float p0, float p1,
                                      Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(max_bin):
            if histo[i]:
                break

        return <dtype_t>(g - i)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_equalize(Py_ssize_t* histo, float pop, dtype_t g,
                                     Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                     float p0, float p1,
                                     Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t sum = 0

    if pop:
        for i in range(max_bin):
            sum += histo[i]
            if i >= g:
                break

        return <dtype_t>(((max_bin - 1) * sum) / pop)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_gradient(Py_ssize_t* histo, float pop, dtype_t g,
                                     Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                     float p0, float p1,
                                     Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i, imin, imax

    if pop:
        for i in range(max_bin - 1, -1, -1):
            if histo[i]:
                imax = i
                break
        for i in range(max_bin):
            if histo[i]:
                imin = i
                break
        return <dtype_t>(imax - imin)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_maximum(Py_ssize_t* histo, float pop, dtype_t g,
                                    Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                    float p0, float p1,
                                    Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(max_bin - 1, -1, -1):
            if histo[i]:
                return <dtype_t>(i)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_mean(Py_ssize_t* histo, float pop, dtype_t g,
                                 Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                 float p0, float p1,
                                 Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(max_bin):
            mean += histo[i] * i
        return <dtype_t>(mean / pop)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_subtract_mean(Py_ssize_t* histo, float pop,
                                          dtype_t g, Py_ssize_t max_bin,
                                          Py_ssize_t mid_bin, float p0,
                                          float p1, Py_ssize_t s0,
                                          Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(max_bin):
            mean += histo[i] * i
        return <dtype_t>((g - mean / pop) / 2. + 127)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_median(Py_ssize_t* histo, float pop, dtype_t g,
                                   Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                   float p0, float p1,
                                   Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef float sum = pop / 2.0

    if pop:
        for i in range(max_bin):
            if histo[i]:
                sum -= histo[i]
                if sum < 0:
                    return <dtype_t>(i)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_minimum(Py_ssize_t* histo, float pop, dtype_t g,
                                    Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                    float p0, float p1,
                                    Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(max_bin):
            if histo[i]:
                return <dtype_t>(i)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_modal(Py_ssize_t* histo, float pop, dtype_t g,
                                  Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                  float p0, float p1,
                                  Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t hmax = 0, imax = 0

    if pop:
        for i in range(max_bin):
            if histo[i] > hmax:
                hmax = histo[i]
                imax = i
        return <dtype_t>(imax)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_enhance_contrast(Py_ssize_t* histo, float pop,
                                             dtype_t g, Py_ssize_t max_bin,
                                             Py_ssize_t mid_bin, float p0,
                                             float p1, Py_ssize_t s0,
                                             Py_ssize_t s1):

    cdef Py_ssize_t i, imin, imax

    if pop:
        for i in range(max_bin - 1, -1, -1):
            if histo[i]:
                imax = i
                break
        for i in range(max_bin):
            if histo[i]:
                imin = i
                break
        if imax - g < g - imin:
            return <dtype_t>(imax)
        else:
            return <dtype_t>(imin)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_pop(Py_ssize_t* histo, float pop, dtype_t g,
                                Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                float p0, float p1,
                                Py_ssize_t s0, Py_ssize_t s1):

    return <dtype_t>(pop)


cdef inline dtype_t _kernel_threshold(Py_ssize_t* histo, float pop, dtype_t g,
                                      Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                      float p0, float p1,
                                      Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(max_bin):
            mean += histo[i] * i
        return <dtype_t>(g > (mean / pop))
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_tophat(Py_ssize_t* histo, float pop, dtype_t g,
                                   Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                   float p0, float p1,
                                   Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(max_bin - 1, -1, -1):
            if histo[i]:
                break

        return <dtype_t>(i - g)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_noise_filter(Py_ssize_t* histo, float pop,
                                         dtype_t g, Py_ssize_t max_bin,
                                         Py_ssize_t mid_bin, float p0, float p1,
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
    for i in range(g, max_bin):
        if histo[i]:
            break
    if i - g < min_i:
        return <dtype_t>(i - g)
    else:
        return <dtype_t>min_i


cdef inline dtype_t _kernel_entropy(Py_ssize_t* histo, float pop, dtype_t g,
                                    Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                    float p0, float p1,
                                    Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef float e, p

    if pop:
        e = 0.

        for i in range(max_bin):
            p = histo[i] / pop
            if p > 0:
                e -= p * log(p) / 0.6931471805599453

        return <dtype_t>e
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_otsu(Py_ssize_t* histo, float pop, dtype_t g,
                                 Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                 float p0, float p1,
                                 Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef Py_ssize_t max_i
    cdef float P, mu1, mu2, q1, new_q1, sigma_b, max_sigma_b
    cdef float mu = 0.

    # compute local mean
    if pop:
        for i in range(max_bin):
            mu += histo[i] * i
        mu = (mu / pop)
    else:
        return <dtype_t>(0)

    # maximizing the between class variance
    max_i = 0
    q1 = histo[0] / pop
    m1 = 0.
    max_sigma_b = 0.

    for i in range(1, max_bin):
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


def _autolevel(dtype_t[:, ::1] image,
               char[:, ::1] selem,
               char[:, ::1] mask,
               dtype_t[:, ::1] out,
               char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_autolevel[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_autolevel[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _bottomhat(dtype_t[:, ::1] image,
               char[:, ::1] selem,
               char[:, ::1] mask,
               dtype_t[:, ::1] out,
               char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_bottomhat[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_bottomhat[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _equalize(dtype_t[:, ::1] image,
              char[:, ::1] selem,
              char[:, ::1] mask,
              dtype_t[:, ::1] out,
              char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_equalize[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_equalize[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _gradient(dtype_t[:, ::1] image,
              char[:, ::1] selem,
              char[:, ::1] mask,
              dtype_t[:, ::1] out,
              char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_gradient[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_gradient[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _maximum(dtype_t[:, ::1] image,
             char[:, ::1] selem,
             char[:, ::1] mask,
             dtype_t[:, ::1] out,
             char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_maximum[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_maximum[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _mean(dtype_t[:, ::1] image,
          char[:, ::1] selem,
          char[:, ::1] mask,
          dtype_t[:, ::1] out,
          char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_mean[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_mean[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _subtract_mean(dtype_t[:, ::1] image,
                   char[:, ::1] selem,
                   char[:, ::1] mask,
                   dtype_t[:, ::1] out,
                   char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_subtract_mean[uint8_t], image, selem, mask,
                       out, shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_subtract_mean[uint16_t], image, selem, mask,
                        out, shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _median(dtype_t[:, ::1] image,
            char[:, ::1] selem,
            char[:, ::1] mask,
            dtype_t[:, ::1] out,
            char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_median[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_median[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _minimum(dtype_t[:, ::1] image,
             char[:, ::1] selem,
             char[:, ::1] mask,
             dtype_t[:, ::1] out,
             char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_minimum[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_minimum[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _enhance_contrast(dtype_t[:, ::1] image,
                      char[:, ::1] selem,
                      char[:, ::1] mask,
                      dtype_t[:, ::1] out,
                      char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_enhance_contrast[uint8_t], image, selem, mask,
                       out, shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_enhance_contrast[uint16_t], image, selem, mask,
                        out, shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _modal(dtype_t[:, ::1] image,
           char[:, ::1] selem,
           char[:, ::1] mask,
           dtype_t[:, ::1] out,
           char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_modal[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_modal[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _pop(dtype_t[:, ::1] image,
         char[:, ::1] selem,
         char[:, ::1] mask,
         dtype_t[:, ::1] out,
         char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_pop[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_pop[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _threshold(dtype_t[:, ::1] image,
               char[:, ::1] selem,
               char[:, ::1] mask,
               dtype_t[:, ::1] out,
               char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_threshold[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_threshold[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _tophat(dtype_t[:, ::1] image,
            char[:, ::1] selem,
            char[:, ::1] mask,
            dtype_t[:, ::1] out,
            char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_tophat[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_tophat[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _noise_filter(dtype_t[:, ::1] image,
                  char[:, ::1] selem,
                  char[:, ::1] mask,
                  dtype_t[:, ::1] out,
                  char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_noise_filter[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_noise_filter[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _entropy(dtype_t[:, ::1] image,
             char[:, ::1] selem,
             char[:, ::1] mask,
             dtype_t[:, ::1] out,
             char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_entropy[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_entropy[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _otsu(dtype_t[:, ::1] image,
          char[:, ::1] selem,
          char[:, ::1] mask,
          dtype_t[:, ::1] out,
          char shift_x, char shift_y, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_otsu[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_otsu[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, 0, 0, max_bin)
