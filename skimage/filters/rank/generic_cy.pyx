#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport log, exp

from .core_cy cimport dtype_t, dtype_t_out, _core

from .core_cy_3d cimport _core_3D

from ..._shared.interpolation cimport round

cnp.import_array()

cdef inline void _kernel_autolevel(dtype_t_out* out, Py_ssize_t odepth,
                                   Py_ssize_t[::1] histo,
                                   double pop, dtype_t g,
                                   Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                   double p0, double p1,
                                   Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i, imin, imax, delta

    if pop:
        for i in range(n_bins - 1, -1, -1):
            if histo[i]:
                imax = i
                break
        for i in range(n_bins):
            if histo[i]:
                imin = i
                break
        delta = imax - imin
        if delta > 0:
            out[0] = <dtype_t_out>((n_bins - 1) * (g - imin) / delta)
        else:
            out[0] = <dtype_t_out>0
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_equalize(dtype_t_out* out, Py_ssize_t odepth,
                                  Py_ssize_t[::1] histo,
                                  double pop, dtype_t g,
                                  Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                  double p0, double p1,
                                  Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t sum = 0

    if pop:
        for i in range(n_bins):
            sum += histo[i]
            if i >= g:
                break
        out[0] = <dtype_t_out>(((n_bins - 1) * sum) / pop)
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_gradient(dtype_t_out* out, Py_ssize_t odepth,
                                  Py_ssize_t[::1] histo,
                                  double pop, dtype_t g,
                                  Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                  double p0, double p1,
                                  Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i, imin, imax

    if pop:
        for i in range(n_bins - 1, -1, -1):
            if histo[i]:
                imax = i
                break
        for i in range(n_bins):
            if histo[i]:
                imin = i
                break
        out[0] = <dtype_t_out>(imax - imin)
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_maximum(dtype_t_out* out, Py_ssize_t odepth,
                                 Py_ssize_t[::1] histo,
                                 double pop, dtype_t g,
                                 Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                 double p0, double p1,
                                 Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i

    if pop:
        for i in range(n_bins - 1, -1, -1):
            if histo[i]:
                out[0] = <dtype_t_out>i
                return
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_mean(dtype_t_out* out, Py_ssize_t odepth,
                              Py_ssize_t[::1] histo,
                              double pop, dtype_t g,
                              Py_ssize_t n_bins, Py_ssize_t mid_bin,
                              double p0, double p1,
                              Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(n_bins):
            mean += histo[i] * i
        out[0] = <dtype_t_out>(mean / pop)
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_geometric_mean(dtype_t_out* out, Py_ssize_t odepth,
                                        Py_ssize_t[::1] histo,
                                        double pop, dtype_t g,
                                        Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                        double p0, double p1,
                                        Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i
    cdef double mean = 0.

    if pop:
        for i in range(n_bins):
            if histo[i]:
                mean += (histo[i] * log(i+1))
        out[0] = <dtype_t_out>round(exp(mean / pop)-1)
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_subtract_mean(dtype_t_out* out, Py_ssize_t odepth,
                                       Py_ssize_t[::1] histo,
                                       double pop, dtype_t g,
                                       Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                       double p0, double p1,
                                       Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(n_bins):
            mean += histo[i] * i
        out[0] = <dtype_t_out>((g - mean / pop) / 2 + mid_bin - 1)
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_median(dtype_t_out* out, Py_ssize_t odepth,
                                Py_ssize_t[::1] histo,
                                double pop, dtype_t g,
                                Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                double p0, double p1,
                                Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i
    cdef double sum = pop / 2.0

    if pop:
        for i in range(n_bins):
            if histo[i]:
                sum -= histo[i]
                if sum < 0:
                    out[0] = <dtype_t_out>i
                    return
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_minimum(dtype_t_out* out, Py_ssize_t odepth,
                                 Py_ssize_t[::1] histo,
                                 double pop, dtype_t g,
                                 Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                 double p0, double p1,
                                 Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i

    if pop:
        for i in range(n_bins):
            if histo[i]:
                out[0] = <dtype_t_out>i
                return
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_modal(dtype_t_out* out, Py_ssize_t odepth,
                               Py_ssize_t[::1] histo,
                               double pop, dtype_t g,
                               Py_ssize_t n_bins, Py_ssize_t mid_bin,
                               double p0, double p1,
                               Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t hmax = 0, imax = 0

    if pop:
        for i in range(n_bins):
            if histo[i] > hmax:
                hmax = histo[i]
                imax = i
        out[0] = <dtype_t_out>imax
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_enhance_contrast(dtype_t_out* out,
                                          Py_ssize_t odepth,
                                          Py_ssize_t[::1] histo,
                                          double pop,
                                          dtype_t g,
                                          Py_ssize_t n_bins,
                                          Py_ssize_t mid_bin, double p0,
                                          double p1, Py_ssize_t s0,
                                          Py_ssize_t s1) nogil:

    cdef Py_ssize_t i, imin, imax

    if pop:
        for i in range(n_bins - 1, -1, -1):
            if histo[i]:
                imax = i
                break
        for i in range(n_bins):
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
                             Py_ssize_t[::1] histo,
                             double pop, dtype_t g,
                             Py_ssize_t n_bins, Py_ssize_t mid_bin,
                             double p0, double p1,
                             Py_ssize_t s0, Py_ssize_t s1) nogil:

    out[0] = <dtype_t_out>pop


cdef inline void _kernel_sum(dtype_t_out* out, Py_ssize_t odepth,
                             Py_ssize_t[::1] histo,
                             double pop, dtype_t g,
                             Py_ssize_t n_bins, Py_ssize_t mid_bin,
                             double p0, double p1,
                             Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t sum = 0

    if pop:
        for i in range(n_bins):
            sum += histo[i] * i
        out[0] = <dtype_t_out>sum
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_threshold(dtype_t_out* out, Py_ssize_t odepth,
                                   Py_ssize_t[::1] histo,
                                   double pop, dtype_t g,
                                   Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                   double p0, double p1,
                                   Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(n_bins):
            mean += histo[i] * i
        out[0] = <dtype_t_out>(g > (mean / pop))
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_noise_filter(dtype_t_out* out, Py_ssize_t odepth,
                                      Py_ssize_t[::1] histo,
                                      double pop, dtype_t g,
                                      Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                      double p0, double p1,
                                      Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t min_i

    # early stop if at least one pixel of the neighborhood has the same g
    if histo[g] > 0:
        out[0] = <dtype_t_out>0
        return

    for i in range(g, -1, -1):
        if histo[i]:
            break
    min_i = g - i
    for i in range(g, n_bins):
        if histo[i]:
            break
    if i - g < min_i:
        out[0] = <dtype_t_out>(i - g)
    else:
        out[0] = <dtype_t_out>min_i


cdef inline void _kernel_entropy(dtype_t_out* out, Py_ssize_t odepth,
                                 Py_ssize_t[::1] histo,
                                 double pop, dtype_t g,
                                 Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                 double p0, double p1,
                                 Py_ssize_t s0, Py_ssize_t s1) nogil:
    cdef Py_ssize_t i
    cdef double e, p

    if pop:
        e = 0.
        for i in range(n_bins):
            p = histo[i] / pop
            if p > 0:
                e -= p * log(p) / 0.6931471805599453
        out[0] = <dtype_t_out>e
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_otsu(dtype_t_out* out, Py_ssize_t odepth,
                              Py_ssize_t[::1] histo,
                              double pop, dtype_t g,
                              Py_ssize_t n_bins, Py_ssize_t mid_bin,
                              double p0, double p1,
                              Py_ssize_t s0, Py_ssize_t s1) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t max_i
    cdef Py_ssize_t P, q1, mu1, mu2, mu = 0
    cdef double sigma_b, max_sigma_b, t

    # compute local mean
    if pop:
        for i in range(n_bins):
            mu += histo[i] * i
    else:
        out[0] = <dtype_t_out>0
        return

    # maximizing the between class variance
    max_i = 0
    q1 = histo[0]
    mu1 = 0
    max_sigma_b = 0.

    for i in range(1, n_bins):
        P = histo[i]
        if P == 0:
            continue

        q1 = q1 + P

        if q1 == pop:
            break

        mu1 = mu1 + i * P
        mu2 = mu - mu1
        t = (pop - q1) * mu1 - mu2 * q1
        sigma_b = (t * t) / (q1 * (pop - q1))
        if sigma_b > max_sigma_b:
            max_sigma_b = sigma_b
            max_i = i

    out[0] = <dtype_t_out>max_i


cdef inline void _kernel_win_hist(dtype_t_out* out, Py_ssize_t odepth,
                                  Py_ssize_t[::1] histo,
                                  double pop, dtype_t g,
                                  Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                  double p0, double p1,
                                  Py_ssize_t s0, Py_ssize_t s1) nogil:
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


cdef inline void _kernel_majority(dtype_t_out* out, Py_ssize_t odepth,
                                  Py_ssize_t[::1] histo,
                                  double pop, dtype_t g,
                                  Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                  double p0, double p1,
                                  Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t votes
    cdef Py_ssize_t candidate = 0

    if pop:
        votes = histo[0]
        for i in range(1, n_bins):
            if histo[i] > votes:
                candidate = i
                votes = histo[i]

    out[0] = <dtype_t_out>(candidate)


def _autolevel(dtype_t[:, ::1] image,
               char[:, ::1] footprint,
               char[:, ::1] mask,
               dtype_t_out[:, :, ::1] out,
               signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_autolevel[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _autolevel_3D(dtype_t[:, :, ::1] image,
                  char[:, :, ::1] footprint,
                  char[:, :, ::1] mask,
                  dtype_t_out[:, :, :, ::1] out,
                  signed char shift_x, signed char shift_y, signed char shift_z,
                  Py_ssize_t n_bins):

    _core_3D(_kernel_autolevel[dtype_t_out, dtype_t], image, footprint, mask,
             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _equalize(dtype_t[:, ::1] image,
              char[:, ::1] footprint,
              char[:, ::1] mask,
              dtype_t_out[:, :, ::1] out,
              signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_equalize[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _equalize_3D(dtype_t[:, :, ::1] image,
                 char[:, :, ::1] footprint,
                 char[:, :, ::1] mask,
                 dtype_t_out[:, :, :, ::1] out,
                 signed char shift_x, signed char shift_y, signed char shift_z,
                 Py_ssize_t n_bins):

    _core_3D(_kernel_equalize[dtype_t_out, dtype_t], image, footprint, mask,
             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _gradient(dtype_t[:, ::1] image,
              char[:, ::1] footprint,
              char[:, ::1] mask,
              dtype_t_out[:, :, ::1] out,
              signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_gradient[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _gradient_3D(dtype_t[:, :, ::1] image,
                 char[:, :, ::1] footprint,
                 char[:, :, ::1] mask,
                 dtype_t_out[:, :, :, ::1] out,
                 signed char shift_x, signed char shift_y, signed char shift_z,
                 Py_ssize_t n_bins):

    _core_3D(_kernel_gradient[dtype_t_out, dtype_t], image, footprint, mask,
             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _maximum(dtype_t[:, ::1] image,
             char[:, ::1] footprint,
             char[:, ::1] mask,
             dtype_t_out[:, :, ::1] out,
             signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_maximum[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _maximum_3D(dtype_t[:, :, ::1] image,
                char[:, :, ::1] footprint,
                char[:, :, ::1] mask,
                dtype_t_out[:, :, :, ::1] out,
                signed char shift_x, signed char shift_y, signed char shift_z,
                Py_ssize_t n_bins):

    _core_3D(_kernel_maximum[dtype_t_out, dtype_t], image, footprint, mask,
             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _mean(dtype_t[:, ::1] image,
          char[:, ::1] footprint,
          char[:, ::1] mask,
          dtype_t_out[:, :, ::1] out,
          signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_mean[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _mean_3D(dtype_t[:, :, ::1] image,
             char[:, :, ::1] footprint,
             char[:, :, ::1] mask,
             dtype_t_out[:, :, :, ::1] out,
             signed char shift_x, signed char shift_y, signed char shift_z,
             Py_ssize_t n_bins):

    _core_3D(_kernel_mean[dtype_t_out, dtype_t], image, footprint, mask, out,
             shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _geometric_mean(dtype_t[:, ::1] image,
                    char[:, ::1] footprint,
                    char[:, ::1] mask,
                    dtype_t_out[:, :, ::1] out,
                    signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_geometric_mean[dtype_t_out, dtype_t], image, footprint, mask,
          out, shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _geometric_mean_3D(dtype_t[:, :, ::1] image,
                       char[:, :, ::1] footprint,
                       char[:, :, ::1] mask,
                       dtype_t_out[:, :, :, ::1] out,
                       signed char shift_x, signed char shift_y, signed char shift_z,
                       Py_ssize_t n_bins):

    _core_3D(_kernel_geometric_mean[dtype_t_out, dtype_t], image, footprint,
             mask, out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _subtract_mean(dtype_t[:, ::1] image,
                   char[:, ::1] footprint,
                   char[:, ::1] mask,
                   dtype_t_out[:, :, ::1] out,
                   signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_subtract_mean[dtype_t_out, dtype_t], image, footprint, mask,
          out, shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _subtract_mean_3D(dtype_t[:, :, ::1] image,
                      char[:, :, ::1] footprint,
                      char[:, :, ::1] mask,
                      dtype_t_out[:, :, :, ::1] out,
                      signed char shift_x, signed char shift_y, signed char shift_z,
                      Py_ssize_t n_bins):

    _core_3D(_kernel_subtract_mean[dtype_t_out, dtype_t], image, footprint,
             mask, out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _median(dtype_t[:, ::1] image,
            char[:, ::1] footprint,
            char[:, ::1] mask,
            dtype_t_out[:, :, ::1] out,
            signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_median[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _median_3D(dtype_t[:, :, ::1] image,
               char[:, :, ::1] footprint,
               char[:, :, ::1] mask,
               dtype_t_out[:, :, :, ::1] out,
               signed char shift_x, signed char shift_y, signed char shift_z,
               Py_ssize_t n_bins):

    _core_3D(_kernel_median[dtype_t_out, dtype_t], image, footprint, mask, out,
             shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _minimum(dtype_t[:, ::1] image,
             char[:, ::1] footprint,
             char[:, ::1] mask,
             dtype_t_out[:, :, ::1] out,
             signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_minimum[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _minimum_3D(dtype_t[:, :, ::1] image,
                char[:, :, ::1] footprint,
                char[:, :, ::1] mask,
                dtype_t_out[:, :, :, ::1] out,
                signed char shift_x, signed char shift_y, signed char shift_z,
                Py_ssize_t n_bins):

    _core_3D(_kernel_minimum[dtype_t_out, dtype_t], image, footprint, mask,
             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _enhance_contrast(dtype_t[:, ::1] image,
                      char[:, ::1] footprint,
                      char[:, ::1] mask,
                      dtype_t_out[:, :, ::1] out,
                      signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_enhance_contrast[dtype_t_out, dtype_t], image, footprint, mask,
          out, shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _enhance_contrast_3D(dtype_t[:, :, ::1] image,
                         char[:, :, ::1] footprint,
                         char[:, :, ::1] mask,
                         dtype_t_out[:, :, :, ::1] out,
                         signed char shift_x, signed char shift_y, signed char shift_z,
                         Py_ssize_t n_bins):

    _core_3D(_kernel_enhance_contrast[dtype_t_out, dtype_t], image, footprint,
             mask, out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _modal(dtype_t[:, ::1] image,
           char[:, ::1] footprint,
           char[:, ::1] mask,
           dtype_t_out[:, :, ::1] out,
           signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_modal[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _modal_3D(dtype_t[:, :, ::1] image,
              char[:, :, ::1] footprint,
              char[:, :, ::1] mask,
              dtype_t_out[:, :, :, ::1] out,
              signed char shift_x, signed char shift_y, signed char shift_z,
              Py_ssize_t n_bins):

    _core_3D(_kernel_modal[dtype_t_out, dtype_t], image, footprint, mask, out,
             shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _pop(dtype_t[:, ::1] image,
         char[:, ::1] footprint,
         char[:, ::1] mask,
         dtype_t_out[:, :, ::1] out,
         signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_pop[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _pop_3D(dtype_t[:, :, ::1] image,
            char[:, :, ::1] footprint,
            char[:, :, ::1] mask,
            dtype_t_out[:, :, :, ::1] out,
            signed char shift_x, signed char shift_y, signed char shift_z,
            Py_ssize_t n_bins):

    _core_3D(_kernel_pop[dtype_t_out, dtype_t], image, footprint, mask, out,
             shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _sum(dtype_t[:, ::1] image,
         char[:, ::1] footprint,
         char[:, ::1] mask,
         dtype_t_out[:, :, ::1] out,
         signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_sum[dtype_t_out, dtype_t], image, footprint, mask,
          out, shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _sum_3D(dtype_t[:, :, ::1] image,
            char[:, :, ::1] footprint,
            char[:, :, ::1] mask,
            dtype_t_out[:, :, :, ::1] out,
            signed char shift_x, signed char shift_y, signed char shift_z,
            Py_ssize_t n_bins):

    _core_3D(_kernel_sum[dtype_t_out, dtype_t], image, footprint, mask, out,
             shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _threshold(dtype_t[:, ::1] image,
               char[:, ::1] footprint,
               char[:, ::1] mask,
               dtype_t_out[:, :, ::1] out,
               signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_threshold[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _threshold_3D(dtype_t[:, :, ::1] image,
                  char[:, :, ::1] footprint,
                  char[:, :, ::1] mask,
                  dtype_t_out[:, :, :, ::1] out,
                  signed char shift_x, signed char shift_y, signed char shift_z,
                  Py_ssize_t n_bins):

    _core_3D(_kernel_threshold[dtype_t_out, dtype_t], image, footprint, mask,
             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _noise_filter(dtype_t[:, ::1] image,
                  char[:, ::1] footprint,
                  char[:, ::1] mask,
                  dtype_t_out[:, :, ::1] out,
                  signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_noise_filter[dtype_t_out, dtype_t], image, footprint, mask,
          out, shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _noise_filter_3D(dtype_t[:, :, ::1] image,
                     char[:, :, ::1] footprint,
                     char[:, :, ::1] mask,
                     dtype_t_out[:, :, :, ::1] out,
                     signed char shift_x, signed char shift_y, signed char shift_z,
                     Py_ssize_t n_bins):

    _core_3D(_kernel_noise_filter[dtype_t_out, dtype_t], image, footprint,
             mask, out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _entropy(dtype_t[:, ::1] image,
             char[:, ::1] footprint,
             char[:, ::1] mask,
             dtype_t_out[:, :, ::1] out,
             signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_entropy[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _entropy_3D(dtype_t[:, :, ::1] image,
                char[:, :, ::1] footprint,
                char[:, :, ::1] mask,
                dtype_t_out[:, :, :, ::1] out,
                signed char shift_x, signed char shift_y, signed char shift_z,
                Py_ssize_t n_bins):

    _core_3D(_kernel_entropy[dtype_t_out, dtype_t], image, footprint, mask,
             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _otsu(dtype_t[:, ::1] image,
          char[:, ::1] footprint,
          char[:, ::1] mask,
          dtype_t_out[:, :, ::1] out,
          signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_otsu[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _otsu_3D(dtype_t[:, :, ::1] image,
             char[:, :, ::1] footprint,
             char[:, :, ::1] mask,
             dtype_t_out[:, :, :, ::1] out,
             signed char shift_x, signed char shift_y, signed char shift_z,
             Py_ssize_t n_bins):

    _core_3D(_kernel_otsu[dtype_t_out, dtype_t], image, footprint, mask, out,
             shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)


def _windowed_hist(dtype_t[:, ::1] image,
                   char[:, ::1] footprint,
                   char[:, ::1] mask,
                   dtype_t_out[:, :, ::1] out,
                   signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_win_hist[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _majority(dtype_t[:, ::1] image,
              char[:, ::1] footprint,
              char[:, ::1] mask,
              dtype_t_out[:, :, ::1] out,
              signed char shift_x, signed char shift_y, Py_ssize_t n_bins):

    _core(_kernel_majority[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, 0, 0, n_bins)


def _majority_3D(dtype_t[:, :, ::1] image,
                 char[:, :, ::1] footprint,
                 char[:, :, ::1] mask,
                 dtype_t_out[:, :, :, ::1] out,
                 signed char shift_x, signed char shift_y, signed char shift_z,
                 Py_ssize_t n_bins):

    _core_3D(_kernel_majority[dtype_t_out, dtype_t], image, footprint, mask,
             out, shift_x, shift_y, shift_z, 0, 0, 0, 0, n_bins)
