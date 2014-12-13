#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport log

from .core_cy cimport dtype_t, dtype_t_out, _core


cdef inline void _kernel_autolevel(dtype_t_out* out, Py_ssize_t odepth,
                                   Py_ssize_t* histo,
                                   double pop, dtype_t g,
                                   Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                   double p0, double p1,
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
            out[0] = <dtype_t_out>((max_bin - 1) * (g - imin) / delta)
        else:
            out[0] = <dtype_t_out>0
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_bottomhat(dtype_t_out* out, Py_ssize_t odepth,
                                   Py_ssize_t* histo,
                                   double pop, dtype_t g,
                                   Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                   double p0, double p1,
                                   Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(max_bin):
            if histo[i]:
                break
        out[0] = <dtype_t_out>(g - i)
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_equalize(dtype_t_out* out, Py_ssize_t odepth,
                                  Py_ssize_t* histo,
                                  double pop, dtype_t g,
                                  Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                  double p0, double p1,
                                  Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t sum = 0

    if pop:
        for i in range(max_bin):
            sum += histo[i]
            if i >= g:
                break
        out[0] = <dtype_t_out>(((max_bin - 1) * sum) / pop)
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_gradient(dtype_t_out* out, Py_ssize_t odepth,
                                  Py_ssize_t* histo,
                                  double pop, dtype_t g,
                                  Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                  double p0, double p1,
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
        out[0] = <dtype_t_out>(imax - imin)
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_maximum(dtype_t_out* out, Py_ssize_t odepth,
                                 Py_ssize_t* histo,
                                 double pop, dtype_t g,
                                 Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                 double p0, double p1,
                                 Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(max_bin - 1, -1, -1):
            if histo[i]:
                out[0] = <dtype_t_out>i
                return
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_mean(dtype_t_out* out, Py_ssize_t odepth,
                              Py_ssize_t* histo,
                              double pop, dtype_t g,
                              Py_ssize_t max_bin, Py_ssize_t mid_bin,
                              double p0, double p1,
                              Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(max_bin):
            mean += histo[i] * i
        out[0] = <dtype_t_out>(mean / pop)
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_subtract_mean(dtype_t_out* out, Py_ssize_t odepth,
                                       Py_ssize_t* histo,
                                       double pop, dtype_t g,
                                       Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                       double p0, double p1,
                                       Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(max_bin):
            mean += histo[i] * i
        out[0] = <dtype_t_out>((g - mean / pop) / 2. + 127)
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_median(dtype_t_out* out, Py_ssize_t odepth,
                                Py_ssize_t* histo,
                                double pop, dtype_t g,
                                Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                double p0, double p1,
                                Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef double sum = pop / 2.0

    if pop:
        for i in range(max_bin):
            if histo[i]:
                sum -= histo[i]
                if sum < 0:
                    out[0] = <dtype_t_out>i
                    return
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_minimum(dtype_t_out* out, Py_ssize_t odepth,
                                 Py_ssize_t* histo,
                                 double pop, dtype_t g,
                                 Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                 double p0, double p1,
                                 Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(max_bin):
            if histo[i]:
                out[0] = <dtype_t_out>i
                return
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_modal(dtype_t_out* out, Py_ssize_t odepth,
                               Py_ssize_t* histo,
                               double pop, dtype_t g,
                               Py_ssize_t max_bin, Py_ssize_t mid_bin,
                               double p0, double p1,
                               Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t hmax = 0, imax = 0

    if pop:
        for i in range(max_bin):
            if histo[i] > hmax:
                hmax = histo[i]
                imax = i
        out[0] = <dtype_t_out>imax
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_enhance_contrast(dtype_t_out* out,
                                          Py_ssize_t odepth,
                                          Py_ssize_t* histo,
                                          double pop,
                                          dtype_t g,
                                          Py_ssize_t max_bin,
                                          Py_ssize_t mid_bin, double p0,
                                          double p1, Py_ssize_t s0,
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
            out[0] = <dtype_t_out>imax
        else:
            out[0] = <dtype_t_out>imin
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_pop(dtype_t_out* out, Py_ssize_t odepth,
                             Py_ssize_t* histo,
                             double pop, dtype_t g,
                             Py_ssize_t max_bin, Py_ssize_t mid_bin,
                             double p0, double p1,
                             Py_ssize_t s0, Py_ssize_t s1):

    out[0] = <dtype_t_out>pop


cdef inline void _kernel_sum(dtype_t_out* out, Py_ssize_t odepth,
                             Py_ssize_t* histo,
                             double pop, dtype_t g,
                             Py_ssize_t max_bin, Py_ssize_t mid_bin,
                             double p0, double p1,
                             Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t sum = 0

    if pop:
        for i in range(max_bin):
            sum += histo[i] * i
        out[0] = <dtype_t_out>sum
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_threshold(dtype_t_out* out, Py_ssize_t odepth,
                                   Py_ssize_t* histo,
                                   double pop, dtype_t g,
                                   Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                   double p0, double p1,
                                   Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(max_bin):
            mean += histo[i] * i
        out[0] = <dtype_t_out>(g > (mean / pop))
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_tophat(dtype_t_out* out, Py_ssize_t odepth,
                                Py_ssize_t* histo,
                                double pop, dtype_t g,
                                Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                double p0, double p1,
                                Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(max_bin - 1, -1, -1):
            if histo[i]:
                break
        out[0] = <dtype_t_out>(i - g)
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_noise_filter(dtype_t_out* out, Py_ssize_t odepth,
                                      Py_ssize_t* histo,
                                      double pop, dtype_t g,
                                      Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                      double p0, double p1,
                                      Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t min_i

    # early stop if at least one pixel of the neighborhood has the same g
    if histo[g] > 0:
        out[0] = <dtype_t_out>0

    for i in range(g, -1, -1):
        if histo[i]:
            break
    min_i = g - i
    for i in range(g, max_bin):
        if histo[i]:
            break
    if i - g < min_i:
        out[0] = <dtype_t_out>(i - g)
    else:
        out[0] = <dtype_t_out>min_i


cdef inline void _kernel_entropy(dtype_t_out* out, Py_ssize_t odepth,
                                 Py_ssize_t* histo,
                                 double pop, dtype_t g,
                                 Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                 double p0, double p1,
                                 Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef double e, p

    if pop:
        e = 0.
        for i in range(max_bin):
            p = histo[i] / pop
            if p > 0:
                e -= p * log(p) / 0.6931471805599453
        out[0] = <dtype_t_out>e
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_otsu(dtype_t_out* out, Py_ssize_t odepth,
                              Py_ssize_t* histo,
                              double pop, dtype_t g,
                              Py_ssize_t max_bin, Py_ssize_t mid_bin,
                              double p0, double p1,
                              Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef Py_ssize_t max_i
    cdef double P, mu1, mu2, q1, new_q1, sigma_b, max_sigma_b
    cdef double mu = 0.

    # compute local mean
    if pop:
        for i in range(max_bin):
            mu += histo[i] * i
        mu = mu / pop
    else:
        out[0] = <dtype_t_out>0

    # maximizing the between class variance
    max_i = 0
    q1 = histo[0] / pop
    mu1 = 0.
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

    out[0] = <dtype_t_out>max_i


cdef inline void _kernel_win_hist(dtype_t_out* out, Py_ssize_t odepth,
                                  Py_ssize_t* histo,
                                  double pop, dtype_t g,
                                  Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                  double p0, double p1,
                                  Py_ssize_t s0, Py_ssize_t s1):
    cdef Py_ssize_t i
    cdef Py_ssize_t max_i
    cdef double scale
    if pop:
        scale = 1.0 / pop
        for i in xrange(odepth):
            out[i] = <dtype_t_out>(histo[i] * scale)
    else:
        for i in xrange(odepth):
            out[i] = <dtype_t_out>0


def _autolevel(dtype_t[:, ::1] image,
               char[:, ::1] selem,
               char[:, ::1] mask,
               dtype_t_out[:, :, ::1] out,
               signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_autolevel[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _bottomhat(dtype_t[:, ::1] image,
               char[:, ::1] selem,
               char[:, ::1] mask,
               dtype_t_out[:, :, ::1] out,
               signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_bottomhat[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _equalize(dtype_t[:, ::1] image,
              char[:, ::1] selem,
              char[:, ::1] mask,
              dtype_t_out[:, :, ::1] out,
              signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_equalize[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _gradient(dtype_t[:, ::1] image,
              char[:, ::1] selem,
              char[:, ::1] mask,
              dtype_t_out[:, :, ::1] out,
              signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_gradient[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _maximum(dtype_t[:, ::1] image,
             char[:, ::1] selem,
             char[:, ::1] mask,
             dtype_t_out[:, :, ::1] out,
             signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_maximum[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _mean(dtype_t[:, ::1] image,
          char[:, ::1] selem,
          char[:, ::1] mask,
          dtype_t_out[:, :, ::1] out,
          signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_mean[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _subtract_mean(dtype_t[:, ::1] image,
                   char[:, ::1] selem,
                   char[:, ::1] mask,
                   dtype_t_out[:, :, ::1] out,
                   signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_subtract_mean[dtype_t_out, dtype_t], image, selem, mask,
          out, shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _median(dtype_t[:, ::1] image,
            char[:, ::1] selem,
            char[:, ::1] mask,
            dtype_t_out[:, :, ::1] out,
            signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_median[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _minimum(dtype_t[:, ::1] image,
             char[:, ::1] selem,
             char[:, ::1] mask,
             dtype_t_out[:, :, ::1] out,
             signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_minimum[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _enhance_contrast(dtype_t[:, ::1] image,
                      char[:, ::1] selem,
                      char[:, ::1] mask,
                      dtype_t_out[:, :, ::1] out,
                      signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_enhance_contrast[dtype_t_out, dtype_t], image, selem, mask,
          out, shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _modal(dtype_t[:, ::1] image,
           char[:, ::1] selem,
           char[:, ::1] mask,
           dtype_t_out[:, :, ::1] out,
           signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_modal[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _pop(dtype_t[:, ::1] image,
         char[:, ::1] selem,
         char[:, ::1] mask,
         dtype_t_out[:, :, ::1] out,
         signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_pop[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _sum(dtype_t[:, ::1] image,
         char[:, ::1] selem,
         char[:, ::1] mask,
         dtype_t_out[:, :, ::1] out,
         signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_sum[dtype_t_out, dtype_t], image, selem, mask,
          out, shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _threshold(dtype_t[:, ::1] image,
               char[:, ::1] selem,
               char[:, ::1] mask,
               dtype_t_out[:, :, ::1] out,
               signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_threshold[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _tophat(dtype_t[:, ::1] image,
            char[:, ::1] selem,
            char[:, ::1] mask,
            dtype_t_out[:, :, ::1] out,
            signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_tophat[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _noise_filter(dtype_t[:, ::1] image,
                  char[:, ::1] selem,
                  char[:, ::1] mask,
                  dtype_t_out[:, :, ::1] out,
                  signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_noise_filter[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _entropy(dtype_t[:, ::1] image,
             char[:, ::1] selem,
             char[:, ::1] mask,
             dtype_t_out[:, :, ::1] out,
             signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_entropy[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _otsu(dtype_t[:, ::1] image,
          char[:, ::1] selem,
          char[:, ::1] mask,
          dtype_t_out[:, :, ::1] out,
          signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_otsu[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _windowed_hist(dtype_t[:, ::1] image,
                   char[:, ::1] selem,
                   char[:, ::1] mask,
                   dtype_t_out[:, :, ::1] out,
                   signed char shift_x, signed char shift_y, Py_ssize_t max_bin):

    _core(_kernel_win_hist[dtype_t_out, dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)
