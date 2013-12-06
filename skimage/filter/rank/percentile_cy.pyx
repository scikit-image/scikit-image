#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from .core_cy cimport dtype_t, dtype_t_out, _core, _min, _max


cdef inline double _kernel_autolevel(Py_ssize_t* histo, double pop, dtype_t g,
                                     Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                     double p0, double p1,
                                     Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i, imin, imax, sum, delta

    if pop:
        sum = 0
        p1 = 1.0 - p1
        for i in range(max_bin):
            sum += histo[i]
            if sum > p0 * pop:
                imin = i
                break
        sum = 0
        for i in range(max_bin - 1, -1, -1):
            sum += histo[i]
            if sum > p1 * pop:
                imax = i
                break

        delta = imax - imin
        if delta > 0:
            return <double>(max_bin - 1) * (_min(_max(imin, g), imax)
                                           - imin) / delta
        else:
            return imax - imin
    else:
        return 0


cdef inline double _kernel_gradient(Py_ssize_t* histo, double pop, dtype_t g,
                                    Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                    double p0, double p1,
                                    Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i, imin, imax, sum, delta

    if pop:
        sum = 0
        p1 = 1.0 - p1
        for i in range(max_bin):
            sum += histo[i]
            if sum >= p0 * pop:
                imin = i
                break
        sum = 0
        for i in range(max_bin - 1, -1, -1):
            sum += histo[i]
            if sum >= p1 * pop:
                imax = i
                break

        return imax - imin
    else:
        return 0


cdef inline double _kernel_mean(Py_ssize_t* histo, double pop, dtype_t g,
                                Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                double p0, double p1,
                                Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i, sum, mean, n

    if pop:
        sum = 0
        mean = 0
        n = 0
        for i in range(max_bin):
            sum += histo[i]
            if (sum >= p0 * pop) and (sum <= p1 * pop):
                n += histo[i]
                mean += histo[i] * i

        if n > 0:
            return mean / n
        else:
            return 0
    else:
        return 0

cdef inline double _kernel_sum(Py_ssize_t* histo, double pop, dtype_t g,
                               Py_ssize_t max_bin, Py_ssize_t mid_bin,
                               double p0, double p1,
                               Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i, sum, sum_g, n

    if pop:
        sum = 0
        sum_g = 0
        n = 0
        for i in range(max_bin):
            sum += histo[i]
            if (sum >= p0 * pop) and (sum <= p1 * pop):
                n += histo[i]
                sum_g += histo[i] * i

        if n > 0:
            return sum_g
        else:
            return 0
    else:
        return 0

cdef inline double _kernel_subtract_mean(Py_ssize_t* histo, double pop,
                                         dtype_t g,
                                         Py_ssize_t max_bin,
                                         Py_ssize_t mid_bin, double p0,
                                         double p1, Py_ssize_t s0,
                                         Py_ssize_t s1):

    cdef Py_ssize_t i, sum, mean, n

    if pop:
        sum = 0
        mean = 0
        n = 0
        for i in range(max_bin):
            sum += histo[i]
            if (sum >= p0 * pop) and (sum <= p1 * pop):
                n += histo[i]
                mean += histo[i] * i
        if n > 0:
            return (g - (mean / n)) * .5 + mid_bin
        else:
            return 0
    else:
        return 0


cdef inline double _kernel_enhance_contrast(Py_ssize_t* histo, double pop,
                                            dtype_t g,
                                            Py_ssize_t max_bin,
                                            Py_ssize_t mid_bin, double p0,
                                            double p1, Py_ssize_t s0,
                                            Py_ssize_t s1):

    cdef Py_ssize_t i, imin, imax, sum, delta

    if pop:
        sum = 0
        p1 = 1.0 - p1
        for i in range(max_bin):
            sum += histo[i]
            if sum > p0 * pop:
                imin = i
                break
        sum = 0
        for i in range(max_bin - 1, -1, -1):
            sum += histo[i]
            if sum > p1 * pop:
                imax = i
                break
        if g > imax:
            return imax
        if g < imin:
            return imin
        if imax - g < g - imin:
            return imax
        else:
            return imin
    else:
        return 0


cdef inline double _kernel_percentile(Py_ssize_t* histo, double pop, dtype_t g,
                                      Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                      double p0, double p1,
                                      Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t sum = 0

    if pop:
        if p0 == 1:  # make sure p0 = 1 returns the maximum filter
            for i in range(max_bin - 1, -1, -1):
                if histo[i]:
                    break
        else:
            for i in range(max_bin):
                sum += histo[i]
                if sum > p0 * pop:
                    break
        return i
    else:
        return 0


cdef inline double _kernel_pop(Py_ssize_t* histo, double pop, dtype_t g,
                               Py_ssize_t max_bin, Py_ssize_t mid_bin,
                               double p0, double p1,
                               Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i, sum, n

    if pop:
        sum = 0
        n = 0
        for i in range(max_bin):
            sum += histo[i]
            if (sum >= p0 * pop) and (sum <= p1 * pop):
                n += histo[i]
        return n
    else:
        return 0


cdef inline double _kernel_threshold(Py_ssize_t* histo, double pop, dtype_t g,
                                     Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                     double p0, double p1,
                                     Py_ssize_t s0, Py_ssize_t s1):

    cdef int i
    cdef Py_ssize_t sum = 0

    if pop:
        for i in range(max_bin):
            sum += histo[i]
            if sum >= p0 * pop:
                break

        return (max_bin - 1) * (g >= i)
    else:
        return 0


def _autolevel(dtype_t[:, ::1] image,
               char[:, ::1] selem,
               char[:, ::1] mask,
               dtype_t_out[:, ::1] out,
               char shift_x, char shift_y, double p0, double p1,
               Py_ssize_t max_bin):

    _core(_kernel_autolevel[dtype_t], image, selem, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, max_bin)


def _gradient(dtype_t[:, ::1] image,
              char[:, ::1] selem,
              char[:, ::1] mask,
              dtype_t_out[:, ::1] out,
              char shift_x, char shift_y, double p0, double p1,
              Py_ssize_t max_bin):

    _core(_kernel_gradient[dtype_t], image, selem, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, max_bin)


def _mean(dtype_t[:, ::1] image,
          char[:, ::1] selem,
          char[:, ::1] mask,
          dtype_t_out[:, ::1] out,
          char shift_x, char shift_y, double p0, double p1,
          Py_ssize_t max_bin):

    _core(_kernel_mean[dtype_t], image, selem, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, max_bin)

def _sum(dtype_t[:, ::1] image,
         char[:, ::1] selem,
         char[:, ::1] mask,
         dtype_t_out[:, ::1] out,
         char shift_x, char shift_y, double p0, double p1,
         Py_ssize_t max_bin):

    _core(_kernel_sum[dtype_t], image, selem, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, max_bin)

def _subtract_mean(dtype_t[:, ::1] image,
                   char[:, ::1] selem,
                   char[:, ::1] mask,
                   dtype_t_out[:, ::1] out,
                   char shift_x, char shift_y, double p0, double p1,
                   Py_ssize_t max_bin):

    _core(_kernel_subtract_mean[dtype_t], image, selem, mask,
          out, shift_x, shift_y, p0, p1, 0, 0, max_bin)


def _enhance_contrast(dtype_t[:, ::1] image,
                      char[:, ::1] selem,
                      char[:, ::1] mask,
                      dtype_t_out[:, ::1] out,
                      char shift_x, char shift_y, double p0, double p1,
                      Py_ssize_t max_bin):

    _core(_kernel_enhance_contrast[dtype_t], image, selem, mask,
          out, shift_x, shift_y, p0, p1, 0, 0, max_bin)


def _percentile(dtype_t[:, ::1] image,
                char[:, ::1] selem,
                char[:, ::1] mask,
                dtype_t_out[:, ::1] out,
                char shift_x, char shift_y, double p0, double p1,
                Py_ssize_t max_bin):

    _core(_kernel_percentile[dtype_t], image, selem, mask, out,
          shift_x, shift_y, p0, 1, 0, 0, max_bin)


def _pop(dtype_t[:, ::1] image,
         char[:, ::1] selem,
         char[:, ::1] mask,
         dtype_t_out[:, ::1] out,
         char shift_x, char shift_y, double p0, double p1,
         Py_ssize_t max_bin):

    _core(_kernel_pop[dtype_t], image, selem, mask, out,
          shift_x, shift_y, p0, p1, 0, 0, max_bin)


def _threshold(dtype_t[:, ::1] image,
               char[:, ::1] selem,
               char[:, ::1] mask,
               dtype_t_out[:, ::1] out,
               char shift_x, char shift_y, double p0, double p1,
               Py_ssize_t max_bin):

    _core(_kernel_threshold[dtype_t], image, selem, mask, out,
          shift_x, shift_y, p0, 1, 0, 0, max_bin)
