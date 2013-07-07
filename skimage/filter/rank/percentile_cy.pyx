#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from .core_cy cimport uint8_t, uint16_t, dtype_t, _core, _min, _max


cdef inline dtype_t _kernel_autolevel(Py_ssize_t* histo, float pop, dtype_t g,
                                      Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                      float p0, float p1,
                                      Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i, imin, imax, sum, delta

    if pop:
        sum = 0
        p1 = 1.0 - p1
        for i in range(max_bin - 1):
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
            return <dtype_t>(<float>(max_bin - 1) * (_min(_max(imin, g), imax)
                             - imin) / delta)
        else:
            return <dtype_t>(imax - imin)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_gradient(Py_ssize_t* histo, float pop, dtype_t g,
                                     Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                     float p0, float p1,
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
        for i in range((max_bin - 1), -1, -1):
            sum += histo[i]
            if sum >= p1 * pop:
                imax = i
                break

        return <dtype_t>(imax - imin)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_mean(Py_ssize_t* histo, float pop, dtype_t g,
                                 Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                 float p0, float p1,
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
            return <dtype_t>(mean / n)
        else:
            return <dtype_t>(0)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_subtract_mean(Py_ssize_t* histo, float pop,
                                          dtype_t g, Py_ssize_t max_bin,
                                          Py_ssize_t mid_bin, float p0,
                                          float p1, Py_ssize_t s0,
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
            return <dtype_t>((g - (mean / n)) * .5 + mid_bin)
        else:
            return <dtype_t>(0)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_enhance_contrast(Py_ssize_t* histo, float pop,
                                             dtype_t g, Py_ssize_t max_bin,
                                             Py_ssize_t mid_bin, float p0,
                                             float p1, Py_ssize_t s0,
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
        for i in range((max_bin - 1), -1, -1):
            sum += histo[i]
            if sum > p1 * pop:
                imax = i
                break
        if g > imax:
            return <dtype_t>imax
        if g < imin:
            return <dtype_t>imin
        if imax - g < g - imin:
            return <dtype_t>imax
        else:
            return <dtype_t>imin
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_percentile(Py_ssize_t* histo, float pop, dtype_t g,
                                       Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                       float p0, float p1,
                                       Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t sum = 0

    if pop:
        for i in range(max_bin):
            sum += histo[i]
            if sum >= p0 * pop:
                break

        return <dtype_t>(i)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_pop(Py_ssize_t* histo, float pop, dtype_t g,
                                Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                float p0, float p1,
                                Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i, sum, n

    if pop:
        sum = 0
        n = 0
        for i in range(max_bin):
            sum += histo[i]
            if (sum >= p0 * pop) and (sum <= p1 * pop):
                n += histo[i]
        return <dtype_t>(n)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_threshold(Py_ssize_t* histo, float pop, dtype_t g,
                                      Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                      float p0, float p1,
                                      Py_ssize_t s0, Py_ssize_t s1):

    cdef int i
    cdef Py_ssize_t sum = 0

    if pop:
        for i in range(max_bin):
            sum += histo[i]
            if sum >= p0 * pop:
                break

        return <dtype_t>((max_bin - 1) * (g >= i))
    else:
        return <dtype_t>(0)


def _autolevel(dtype_t[:, ::1] image,
              char[:, ::1] selem,
              char[:, ::1] mask,
              dtype_t[:, ::1] out,
              char shift_x, char shift_y, float p0, float p1,
              Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_autolevel[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, p0, p1, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_autolevel[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, p0, p1, 0, 0, max_bin)


def _gradient(dtype_t[:, ::1] image,
             char[:, ::1] selem,
             char[:, ::1] mask,
             dtype_t[:, ::1] out,
             char shift_x, char shift_y, float p0, float p1,
             Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_gradient[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, p0, p1, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_gradient[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, p0, p1, 0, 0, max_bin)


def _mean(dtype_t[:, ::1] image,
         char[:, ::1] selem,
         char[:, ::1] mask,
         dtype_t[:, ::1] out,
         char shift_x, char shift_y, float p0, float p1,
         Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_mean[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, p0, p1, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_mean[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, p0, p1, 0, 0, max_bin)


def _subtract_mean(dtype_t[:, ::1] image,
                   char[:, ::1] selem,
                   char[:, ::1] mask,
                   dtype_t[:, ::1] out,
                   char shift_x, char shift_y, float p0, float p1,
                   Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_subtract_mean[uint8_t], image, selem, mask,
                       out, shift_x, shift_y, p0, p1, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_subtract_mean[uint16_t], image, selem, mask,
                        out, shift_x, shift_y, p0, p1, 0, 0, max_bin)


def _enhance_contrast(dtype_t[:, ::1] image,
                    char[:, ::1] selem,
                    char[:, ::1] mask,
                    dtype_t[:, ::1] out,
                    char shift_x, char shift_y, float p0, float p1,
                    Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_enhance_contrast[uint8_t], image, selem, mask,
                       out, shift_x, shift_y, p0, p1, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_enhance_contrast[uint16_t], image, selem, mask,
                        out, shift_x, shift_y, p0, p1, 0, 0, max_bin)


def _percentile(dtype_t[:, ::1] image,
               char[:, ::1] selem,
               char[:, ::1] mask,
               dtype_t[:, ::1] out,
               char shift_x, char shift_y, float p0, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_percentile[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, p0, 1, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_percentile[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, p0, 1, 0, 0, max_bin)


def _pop(dtype_t[:, ::1] image,
        char[:, ::1] selem,
        char[:, ::1] mask,
        dtype_t[:, ::1] out,
        char shift_x, char shift_y, float p0, float p1,
        Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_pop[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, p0, p1, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_pop[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, p0, p1, 0, 0, max_bin)


def _threshold(dtype_t[:, ::1] image,
              char[:, ::1] selem,
              char[:, ::1] mask,
              dtype_t[:, ::1] out,
              char shift_x, char shift_y, float p0, Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_threshold[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, p0, 1, 0, 0, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_threshold[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, p0, 1, 0, 0, max_bin)
