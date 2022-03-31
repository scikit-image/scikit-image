#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from .core_cy cimport dtype_t, dtype_t_out, _core, _min, _max
cnp.import_array()

cdef inline void _kernel_autolevel(dtype_t_out* out, Py_ssize_t odepth,
                                   Py_ssize_t[::1] histo,
                                   double pop, dtype_t g,
                                   Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                   double p0, double p1,
                                   Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i, imin, imax, sum, delta

    if pop:
        sum = 0
        p1 = 1.0 - p1
        for i in range(n_bins):
            sum += histo[i]
            if sum > p0 * pop:
                imin = i
                break
        sum = 0
        for i in range(n_bins - 1, -1, -1):
            sum += histo[i]
            if sum > p1 * pop:
                imax = i
                break

        delta = imax - imin
        if delta > 0:
            out[0] = <dtype_t_out>((n_bins - 1) * (_min(_max(imin, g), imax)
                                           - imin) / delta)
        else:
            out[0] = <dtype_t_out>(imax - imin)
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_gradient(dtype_t_out* out, Py_ssize_t odepth,
                                  Py_ssize_t[::1] histo,
                                  double pop, dtype_t g,
                                  Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                  double p0, double p1,
                                  Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i, imin, imax, sum, delta

    if pop:
        sum = 0
        p1 = 1.0 - p1
        for i in range(n_bins):
            sum += histo[i]
            if sum >= p0 * pop:
                imin = i
                break
        sum = 0
        for i in range(n_bins - 1, -1, -1):
            sum += histo[i]
            if sum >= p1 * pop:
                imax = i
                break

        out[0] = <dtype_t_out>(imax - imin)
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_mean(dtype_t_out* out, Py_ssize_t odepth,
                              Py_ssize_t[::1] histo,
                              double pop, dtype_t g,
                              Py_ssize_t n_bins, Py_ssize_t mid_bin,
                              double p0, double p1,
                              Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i, sum, mean, n

    if pop:
        sum = 0
        mean = 0
        n = 0
        for i in range(n_bins):
            sum += histo[i]
            if (sum >= p0 * pop) and (sum <= p1 * pop):
                n += histo[i]
                mean += histo[i] * i

        if n > 0:
            out[0] = <dtype_t_out>(mean / n)
        else:
            out[0] = <dtype_t_out>0
    else:
        out[0] = <dtype_t_out>0

cdef inline void _kernel_sum(dtype_t_out* out, Py_ssize_t odepth,
                             Py_ssize_t[::1] histo,
                             double pop, dtype_t g,
                             Py_ssize_t n_bins, Py_ssize_t mid_bin,
                             double p0, double p1,
                             Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i, sum, sum_g, n

    if pop:
        sum = 0
        sum_g = 0
        n = 0
        for i in range(n_bins):
            sum += histo[i]
            if (sum >= p0 * pop) and (sum <= p1 * pop):
                n += histo[i]
                sum_g += histo[i] * i

        if n > 0:
            out[0] = <dtype_t_out>sum_g
        else:
            out[0] = <dtype_t_out>0
    else:
        out[0] = <dtype_t_out>0

cdef inline void _kernel_subtract_mean(dtype_t_out* out, Py_ssize_t odepth,
                                       Py_ssize_t[::1] histo,
                                       double pop, dtype_t g,
                                       Py_ssize_t n_bins,
                                       Py_ssize_t mid_bin, double p0,
                                       double p1, Py_ssize_t s0,
                                       Py_ssize_t s1) nogil:

    cdef Py_ssize_t i, sum, mean, n

    if pop:
        sum = 0
        mean = 0
        n = 0
        for i in range(n_bins):
            sum += histo[i]
            if (sum >= p0 * pop) and (sum <= p1 * pop):
                n += histo[i]
                mean += histo[i] * i
        if n > 0:
            out[0] = <dtype_t_out>((g - (mean / n)) * .5 + mid_bin)
        else:
            out[0] = <dtype_t_out>0
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_enhance_contrast(dtype_t_out* out,
                                          Py_ssize_t odepth,
                                          Py_ssize_t[::1] histo, double pop,
                                          dtype_t g,
                                          Py_ssize_t n_bins,
                                          Py_ssize_t mid_bin, double p0,
                                          double p1, Py_ssize_t s0,
                                          Py_ssize_t s1) nogil:

    cdef Py_ssize_t i, imin, imax, sum, delta

    if pop:
        sum = 0
        p1 = 1.0 - p1
        for i in range(n_bins):
            sum += histo[i]
            if sum > p0 * pop:
                imin = i
                break
        sum = 0
        for i in range(n_bins - 1, -1, -1):
            sum += histo[i]
            if sum > p1 * pop:
                imax = i
                break
        if g > imax:
            out[0] = <dtype_t_out>imax
        if g < imin:
            out[0] = <dtype_t_out>imin
        if imax - g < g - imin:
            out[0] = <dtype_t_out>imax
        else:
            out[0] = <dtype_t_out>imin
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_percentile(dtype_t_out* out, Py_ssize_t odepth,
                                    Py_ssize_t[::1] histo,
                                    double pop, dtype_t g,
                                    Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                    double p0, double p1,
                                    Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t sum = 0

    if pop:
        if p0 == 1:  # make sure p0 = 1 returns the maximum filter
            for i in range(n_bins - 1, -1, -1):
                if histo[i]:
                    break
        else:
            for i in range(n_bins):
                sum += histo[i]
                if sum > p0 * pop:
                    break
        out[0] = <dtype_t_out>i
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_pop(dtype_t_out* out, Py_ssize_t odepth,
                             Py_ssize_t[::1] histo,
                             double pop, dtype_t g,
                             Py_ssize_t n_bins, Py_ssize_t mid_bin,
                             double p0, double p1,
                             Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i, sum, n

    if pop:
        sum = 0
        n = 0
        for i in range(n_bins):
            sum += histo[i]
            if (sum >= p0 * pop) and (sum <= p1 * pop):
                n += histo[i]
        out[0] = <dtype_t_out>n
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_threshold(dtype_t_out* out, Py_ssize_t odepth,
                                   Py_ssize_t[::1] histo,
                                   double pop, dtype_t g,
                                   Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                   double p0, double p1,
                                   Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef int i
    cdef Py_ssize_t sum = 0

    if pop:
        for i in range(n_bins):
            sum += histo[i]
            if sum >= p0 * pop:
                break

        out[0] = <dtype_t_out>((n_bins - 1) * (g >= i))
    else:
        out[0] = <dtype_t_out>0


def _autolevel(dtype_t[:, ::1] image,
               char[:, ::1] footprint,
               char[:, ::1] mask,
               dtype_t_out[:, :, ::1] out,
               signed char shift_x, signed char shift_y, double p0, double p1,
               Py_ssize_t n_bins):

    _core(_kernel_autolevel[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _gradient(dtype_t[:, ::1] image,
              char[:, ::1] footprint,
              char[:, ::1] mask,
              dtype_t_out[:, :, ::1] out,
              signed char shift_x, signed char shift_y, double p0, double p1,
              Py_ssize_t n_bins):

    _core(_kernel_gradient[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _mean(dtype_t[:, ::1] image,
          char[:, ::1] footprint,
          char[:, ::1] mask,
          dtype_t_out[:, :, ::1] out,
          signed char shift_x, signed char shift_y, double p0, double p1,
          Py_ssize_t n_bins):

    _core(_kernel_mean[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _sum(dtype_t[:, ::1] image,
         char[:, ::1] footprint,
         char[:, ::1] mask,
         dtype_t_out[:, :, ::1] out,
         signed char shift_x, signed char shift_y, double p0, double p1,
         Py_ssize_t n_bins):

    _core(_kernel_sum[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _subtract_mean(dtype_t[:, ::1] image,
                   char[:, ::1] footprint,
                   char[:, ::1] mask,
                   dtype_t_out[:, :, ::1] out,
                   signed char shift_x, signed char shift_y, double p0, double p1,
                   Py_ssize_t n_bins):

    _core(_kernel_subtract_mean[dtype_t_out, dtype_t], image, footprint, mask,
          out, shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _enhance_contrast(dtype_t[:, ::1] image,
                      char[:, ::1] footprint,
                      char[:, ::1] mask,
                      dtype_t_out[:, :, ::1] out,
                      signed char shift_x, signed char shift_y, double p0, double p1,
                      Py_ssize_t n_bins):

    _core(_kernel_enhance_contrast[dtype_t_out, dtype_t], image, footprint,
          mask, out, shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _percentile(dtype_t[:, ::1] image,
                char[:, ::1] footprint,
                char[:, ::1] mask,
                dtype_t_out[:, :, ::1] out,
                signed char shift_x, signed char shift_y, double p0, double p1,
                Py_ssize_t n_bins):

    _core(_kernel_percentile[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, 1, 0, 0, n_bins)


def _pop(dtype_t[:, ::1] image,
         char[:, ::1] footprint,
         char[:, ::1] mask,
         dtype_t_out[:, :, ::1] out,
         signed char shift_x, signed char shift_y, double p0, double p1,
         Py_ssize_t n_bins):

    _core(_kernel_pop[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _threshold(dtype_t[:, ::1] image,
               char[:, ::1] footprint,
               char[:, ::1] mask,
               dtype_t_out[:, :, ::1] out,
               signed char shift_x, signed char shift_y, double p0, double p1,
               Py_ssize_t n_bins):

    _core(_kernel_threshold[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, 1, 0, 0, n_bins)
