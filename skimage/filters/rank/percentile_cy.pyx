#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from .core_cy cimport dtype_t, dtype_t_out, _core, _min, _max
from libc.math cimport floor, ceil
cnp.import_array()

cdef inline void _kernel_autolevel(dtype_t_out* out, Py_ssize_t odepth,
                                   Py_ssize_t[::1] histo,
                                   cnp.float64_t pop, dtype_t g,
                                   Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                   cnp.float64_t p0, cnp.float64_t p1,
                                   Py_ssize_t s0, Py_ssize_t s1) noexcept nogil:

    cdef Py_ssize_t i = 0, imin = 0, imax = 0, sum, delta

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
                                  cnp.float64_t pop, dtype_t g,
                                  Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                  cnp.float64_t p0, cnp.float64_t p1,
                                  Py_ssize_t s0, Py_ssize_t s1) noexcept nogil:

    cdef Py_ssize_t i, imin = 0, imax = 0, sum

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
                              cnp.float64_t pop, dtype_t g,
                              Py_ssize_t n_bins, Py_ssize_t mid_bin,
                              cnp.float64_t p0, cnp.float64_t p1,
                              Py_ssize_t s0, Py_ssize_t s1) noexcept nogil:
    """Return local mean of an histogram excluding optional outer percentiles.

    This algorithm uses two counters -- `lower` and `inner``-- to average the
    appropriate bins of the histogram. First, the number of pixels in each excluded bin
    are subtracted from `lower` until the percentile is reached for which the arithmetic
    mean is to be calculated. The same is repeated with the `inner` counter, while
    summing the value of pixels in each bin which is later divided by the total number
    of included pixels.
    """
    cdef:
        Py_ssize_t i, lower, inner, denominator
        cnp.uint64_t total = 0

    # Counter to deplete while summing lower excluded bins in histogram
    lower = <Py_ssize_t>ceil(pop * p0)
    # Safely subtract 1 because border should be included in average
    if 0 < lower:
        lower -= 1
    # Counter to deplete while summing inner included bins in histogram
    inner = <Py_ssize_t>floor(pop * p1)
    # Safely add 1 because border should be included in average
    if inner < pop:
        inner += 1
    # Inner counter starts after `lower` is depleted, so adjust
    inner -= lower
    denominator = inner

    if denominator <= 0:
        out[0] = <dtype_t_out> 0
        return  # Return early

    i = 0
    # Deplete counter `lower` by subtracting lower excluded bins
    while 0 < lower:
        lower -= histo[i]
        i += 1
    # If `lower` is negative, percentile border is inside bin
    # so add as many pixel values as `lower` underflowed
    total += -lower * (i - 1)
    inner += lower
    # Deplete counter `inner` by subtracting inner included bins, upper excluded bins
    # are therefore implicitly excluded
    while 0 < inner:
        inner -= histo[i]
        total += histo[i] * i
        i += 1
    # If `inner` is negative, percentile border is inside bin, and we added too
    # much, so subtract as many pixel values as `inner` underflowed
    total += inner * (i - 1)
    # Drop remainder of mean to maintain backwards compatibility
    out[0] = <dtype_t_out>(total / denominator)


cdef inline void _kernel_sum(dtype_t_out* out, Py_ssize_t odepth,
                             Py_ssize_t[::1] histo,
                             cnp.float64_t pop, dtype_t g,
                             Py_ssize_t n_bins, Py_ssize_t mid_bin,
                             cnp.float64_t p0, cnp.float64_t p1,
                             Py_ssize_t s0, Py_ssize_t s1) noexcept nogil:

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
                                       cnp.float64_t pop, dtype_t g,
                                       Py_ssize_t n_bins,
                                       Py_ssize_t mid_bin, cnp.float64_t p0,
                                       cnp.float64_t p1, Py_ssize_t s0,
                                       Py_ssize_t s1) noexcept nogil:

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
                                          Py_ssize_t[::1] histo, cnp.float64_t pop,
                                          dtype_t g,
                                          Py_ssize_t n_bins,
                                          Py_ssize_t mid_bin, cnp.float64_t p0,
                                          cnp.float64_t p1, Py_ssize_t s0,
                                          Py_ssize_t s1) noexcept nogil:

    cdef Py_ssize_t i, imin = 0, imax = 0, sum

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
                                    cnp.float64_t pop, dtype_t g,
                                    Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                    cnp.float64_t p0, cnp.float64_t p1,
                                    Py_ssize_t s0, Py_ssize_t s1) noexcept nogil:

    cdef Py_ssize_t i = 0
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
                             cnp.float64_t pop, dtype_t g,
                             Py_ssize_t n_bins, Py_ssize_t mid_bin,
                             cnp.float64_t p0, cnp.float64_t p1,
                             Py_ssize_t s0, Py_ssize_t s1) noexcept nogil:

    cdef Py_ssize_t i = 0, sum, n

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
                                   cnp.float64_t pop, dtype_t g,
                                   Py_ssize_t n_bins, Py_ssize_t mid_bin,
                                   cnp.float64_t p0, cnp.float64_t p1,
                                   Py_ssize_t s0, Py_ssize_t s1) noexcept nogil:

    cdef int i = 0
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
               signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
               Py_ssize_t n_bins):

    _core(_kernel_autolevel[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _gradient(dtype_t[:, ::1] image,
              char[:, ::1] footprint,
              char[:, ::1] mask,
              dtype_t_out[:, :, ::1] out,
              signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
              Py_ssize_t n_bins):

    _core(_kernel_gradient[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _mean(dtype_t[:, ::1] image,
          char[:, ::1] footprint,
          char[:, ::1] mask,
          dtype_t_out[:, :, ::1] out,
          signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
          Py_ssize_t n_bins):

    _core(_kernel_mean[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _sum(dtype_t[:, ::1] image,
         char[:, ::1] footprint,
         char[:, ::1] mask,
         dtype_t_out[:, :, ::1] out,
         signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
         Py_ssize_t n_bins):

    _core(_kernel_sum[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _subtract_mean(dtype_t[:, ::1] image,
                   char[:, ::1] footprint,
                   char[:, ::1] mask,
                   dtype_t_out[:, :, ::1] out,
                   signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
                   Py_ssize_t n_bins):

    _core(_kernel_subtract_mean[dtype_t_out, dtype_t], image, footprint, mask,
          out, shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _enhance_contrast(dtype_t[:, ::1] image,
                      char[:, ::1] footprint,
                      char[:, ::1] mask,
                      dtype_t_out[:, :, ::1] out,
                      signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
                      Py_ssize_t n_bins):

    _core(_kernel_enhance_contrast[dtype_t_out, dtype_t], image, footprint,
          mask, out, shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _percentile(dtype_t[:, ::1] image,
                char[:, ::1] footprint,
                char[:, ::1] mask,
                dtype_t_out[:, :, ::1] out,
                signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
                Py_ssize_t n_bins):

    _core(_kernel_percentile[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, 1, 0, 0, n_bins)


def _pop(dtype_t[:, ::1] image,
         char[:, ::1] footprint,
         char[:, ::1] mask,
         dtype_t_out[:, :, ::1] out,
         signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
         Py_ssize_t n_bins):

    _core(_kernel_pop[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, n_bins)


def _threshold(dtype_t[:, ::1] image,
               char[:, ::1] footprint,
               char[:, ::1] mask,
               dtype_t_out[:, :, ::1] out,
               signed char shift_x, signed char shift_y, cnp.float64_t p0, cnp.float64_t p1,
               Py_ssize_t n_bins):

    _core(_kernel_threshold[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, p0, 1, 0, 0, n_bins)
