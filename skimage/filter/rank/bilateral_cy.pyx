#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport log

from .core_cy cimport dtype_t, dtype_t_out, _core


cdef inline double _kernel_mean(Py_ssize_t* histo, double pop, dtype_t g,
                                Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                double p0, double p1,
                                Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t bilat_pop = 0
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(max_bin):
            if (g > (i - s0)) and (g < (i + s1)):
                bilat_pop += histo[i]
                mean += histo[i] * i
        if bilat_pop:
            return mean / bilat_pop
        else:
            return 0
    else:
        return 0


cdef inline double _kernel_pop(Py_ssize_t* histo, double pop, dtype_t g,
                               Py_ssize_t max_bin, Py_ssize_t mid_bin,
                               double p0, double p1,
                               Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t bilat_pop = 0

    if pop:
        for i in range(max_bin):
            if (g > (i - s0)) and (g < (i + s1)):
                bilat_pop += histo[i]
        return bilat_pop
    else:
        return 0

cdef inline double _kernel_sum(Py_ssize_t* histo, double pop, dtype_t g,
                                Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                double p0, double p1,
                                Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t bilat_pop = 0
    cdef Py_ssize_t sum = 0

    if pop:
        for i in range(max_bin):
            if (g > (i - s0)) and (g < (i + s1)):
                bilat_pop += histo[i]
                sum += histo[i] * i
        if bilat_pop:
            return sum
        else:
            return 0
    else:
        return 0


def _mean(dtype_t[:, ::1] image,
          char[:, ::1] selem,
          char[:, ::1] mask,
          dtype_t_out[:, ::1] out,
          char shift_x, char shift_y, Py_ssize_t s0, Py_ssize_t s1,
          Py_ssize_t max_bin):

    _core(_kernel_mean[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, s0, s1, max_bin)


def _pop(dtype_t[:, ::1] image,
         char[:, ::1] selem,
         char[:, ::1] mask,
         dtype_t_out[:, ::1] out,
         char shift_x, char shift_y, Py_ssize_t s0, Py_ssize_t s1,
         Py_ssize_t max_bin):

    _core(_kernel_pop[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, s0, s1, max_bin)

def _sum(dtype_t[:, ::1] image,
         char[:, ::1] selem,
         char[:, ::1] mask,
         dtype_t_out[:, ::1] out,
         char shift_x, char shift_y, Py_ssize_t s0, Py_ssize_t s1,
         Py_ssize_t max_bin):

    _core(_kernel_sum[dtype_t], image, selem, mask, out,
          shift_x, shift_y, 0, 0, s0, s1, max_bin)
