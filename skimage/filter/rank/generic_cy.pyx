#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport log

from .core_cy cimport dtype_t, dtype_t_out, _core


cdef inline double _kernel_autolevel(Py_ssize_t* histo, double pop, dtype_t g,
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
            return <double>(max_bin - 1) * (g - imin) / delta
        else:
            return 0
    else:
        return 0


cdef inline double _kernel_bottomhat(Py_ssize_t* histo, double pop, dtype_t g,
                                     Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                     double p0, double p1,
                                     Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(max_bin):
            if histo[i]:
                break
        return g - i
    else:
        return 0


cdef inline double _kernel_equalize(Py_ssize_t* histo, double pop, dtype_t g,
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
        return ((max_bin - 1) * sum) / pop
    else:
        return 0


cdef inline double _kernel_gradient(Py_ssize_t* histo, double pop, dtype_t g,
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
        return imax - imin
    else:
        return 0


cdef inline double _kernel_maximum(Py_ssize_t* histo, double pop, dtype_t g,
                                   Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                   double p0, double p1,
                                   Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(max_bin - 1, -1, -1):
            if histo[i]:
                return i
    else:
        return 0


cdef inline double _kernel_mean(Py_ssize_t* histo, double pop,dtype_t g,
                                Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                double p0, double p1,
                                Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(max_bin):
            mean += histo[i] * i
        return mean / pop
    else:
        return 0


cdef inline double _kernel_subtract_mean(Py_ssize_t* histo, double pop,
                                         dtype_t g,
                                         Py_ssize_t max_bin,
                                         Py_ssize_t mid_bin, double p0,
                                         double p1, Py_ssize_t s0,
                                         Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(max_bin):
            mean += histo[i] * i
        return (g - mean / pop) / 2. + 127
    else:
        return 0


cdef inline double _kernel_median(Py_ssize_t* histo, double pop, dtype_t g,
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
                    return i
    else:
        return 0


cdef inline double _kernel_minimum(Py_ssize_t* histo, double pop, dtype_t g,
                                   Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                   double p0, double p1,
                                   Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(max_bin):
            if histo[i]:
                return i
    else:
        return 0


cdef inline double _kernel_modal(Py_ssize_t* histo, double pop, dtype_t g,
                                 Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                 double p0, double p1,
                                 Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t hmax = 0, imax = 0

    if pop:
        for i in range(max_bin):
            if histo[i] > hmax:
                hmax = histo[i]
                imax = i
        return imax
    else:
        return 0


cdef inline double _kernel_enhance_contrast(Py_ssize_t* histo, double pop,
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
            return imax
        else:
            return imin
    else:
        return 0


cdef inline double _kernel_pop(Py_ssize_t* histo, double pop, dtype_t g,
                               Py_ssize_t max_bin, Py_ssize_t mid_bin,
                               double p0, double p1,
                               Py_ssize_t s0, Py_ssize_t s1):

    return pop


cdef inline double _kernel_sum(Py_ssize_t* histo, double pop,dtype_t g,
                               Py_ssize_t max_bin, Py_ssize_t mid_bin,
                               double p0, double p1,
                               Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t sum = 0

    if pop:
        for i in range(max_bin):
            sum += histo[i] * i
        return sum
    else:
        return 0


cdef inline double _kernel_threshold(Py_ssize_t* histo, double pop, dtype_t g,
                                     Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                     double p0, double p1,
                                     Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(max_bin):
            mean += histo[i] * i
        return g > (mean / pop)
    else:
        return 0


cdef inline double _kernel_tophat(Py_ssize_t* histo, double pop, dtype_t g,
                                  Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                  double p0, double p1,
                                  Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i

    if pop:
        for i in range(max_bin - 1, -1, -1):
            if histo[i]:
                break
        return i - g
    else:
        return 0


cdef inline double _kernel_noise_filter(Py_ssize_t* histo, double pop,
                                        dtype_t g, Py_ssize_t max_bin,
                                        Py_ssize_t mid_bin, double p0,
                                        double p1, Py_ssize_t s0,
                                        Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t min_i

    # early stop if at least one pixel of the neighborhood has the same g
    if histo[g] > 0:
        return 0

    for i in range(g, -1, -1):
        if histo[i]:
            break
    min_i = g - i
    for i in range(g, max_bin):
        if histo[i]:
            break
    if i - g < min_i:
        return i - g
    else:
        return min_i


cdef inline double _kernel_entropy(Py_ssize_t* histo, double pop, dtype_t g,
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
        return e
    else:
        return 0


cdef inline double _kernel_otsu(Py_ssize_t* histo, double pop, dtype_t g,
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
        return 0

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

    return max_i


def _autolevel(dtype_t[:, ::1] image,
               char[:, ::1] selem,
               char[:, ::1] mask,
               dtype_t_out[:, ::1] out,
               char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_autolevel[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _bottomhat(dtype_t[:, ::1] image,
               char[:, ::1] selem,
               char[:, ::1] mask,
               dtype_t_out[:, ::1] out,
               char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_bottomhat[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _equalize(dtype_t[:, ::1] image,
              char[:, ::1] selem,
              char[:, ::1] mask,
              dtype_t_out[:, ::1] out,
              char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_equalize[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _gradient(dtype_t[:, ::1] image,
              char[:, ::1] selem,
              char[:, ::1] mask,
              dtype_t_out[:, ::1] out,
              char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_gradient[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _maximum(dtype_t[:, ::1] image,
             char[:, ::1] selem,
             char[:, ::1] mask,
             dtype_t_out[:, ::1] out,
             char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_maximum[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _mean(dtype_t[:, ::1] image,
          char[:, ::1] selem,
          char[:, ::1] mask,
          dtype_t_out[:, ::1] out,
          char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_mean[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _subtract_mean(dtype_t[:, ::1] image,
                   char[:, ::1] selem,
                   char[:, ::1] mask,
                   dtype_t_out[:, ::1] out,
                   char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_subtract_mean[dtype_t], image, selem, mask,
          out, shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _median(dtype_t[:, ::1] image,
            char[:, ::1] selem,
            char[:, ::1] mask,
            dtype_t_out[:, ::1] out,
            char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_median[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _minimum(dtype_t[:, ::1] image,
             char[:, ::1] selem,
             char[:, ::1] mask,
             dtype_t_out[:, ::1] out,
             char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_minimum[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _enhance_contrast(dtype_t[:, ::1] image,
                      char[:, ::1] selem,
                      char[:, ::1] mask,
                      dtype_t_out[:, ::1] out,
                      char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_enhance_contrast[dtype_t], image, selem, mask,
          out, shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _modal(dtype_t[:, ::1] image,
           char[:, ::1] selem,
           char[:, ::1] mask,
           dtype_t_out[:, ::1] out,
           char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_modal[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _pop(dtype_t[:, ::1] image,
         char[:, ::1] selem,
         char[:, ::1] mask,
         dtype_t_out[:, ::1] out,
         char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_pop[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)

def _sum(dtype_t[:, ::1] image,
         char[:, ::1] selem,
         char[:, ::1] mask,
         dtype_t_out[:, ::1] out,
         char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_sum[dtype_t], image, selem, mask,
          out, shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _threshold(dtype_t[:, ::1] image,
               char[:, ::1] selem,
               char[:, ::1] mask,
               dtype_t_out[:, ::1] out,
               char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_threshold[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _tophat(dtype_t[:, ::1] image,
            char[:, ::1] selem,
            char[:, ::1] mask,
            dtype_t_out[:, ::1] out,
            char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_tophat[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _noise_filter(dtype_t[:, ::1] image,
                  char[:, ::1] selem,
                  char[:, ::1] mask,
                  dtype_t_out[:, ::1] out,
                  char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_noise_filter[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _entropy(dtype_t[:, ::1] image,
             char[:, ::1] selem,
             char[:, ::1] mask,
             dtype_t_out[:, ::1] out,
             char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_entropy[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)


def _otsu(dtype_t[:, ::1] image,
          char[:, ::1] selem,
          char[:, ::1] mask,
          dtype_t_out[:, ::1] out,
          char shift_x, char shift_y, Py_ssize_t max_bin):

    _core(_kernel_otsu[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, max_bin)
