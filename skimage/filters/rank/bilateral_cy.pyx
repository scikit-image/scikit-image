#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport log

from .core_cy cimport dtype_t, dtype_t_out, _core

cnp.import_array()

cdef inline void _kernel_mean(dtype_t_out* out, Py_ssize_t odepth,
                              Py_ssize_t[::1] histo,
                              double pop, dtype_t g,
                              Py_ssize_t n_bins, Py_ssize_t mid_bin,
                              double p0, double p1,
                              Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t bilat_pop = 0
    cdef Py_ssize_t mean = 0

    if pop:
        for i in range(n_bins):
            if (g > (i - s0)) and (g < (i + s1)):
                bilat_pop += histo[i]
                mean += histo[i] * i
        if bilat_pop:
            out[0] = <dtype_t_out>(mean / bilat_pop)
        else:
            out[0] = <dtype_t_out>0
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_pop(dtype_t_out* out, Py_ssize_t odepth,
                             Py_ssize_t[::1] histo,
                             double pop, dtype_t g,
                             Py_ssize_t n_bins, Py_ssize_t mid_bin,
                             double p0, double p1,
                             Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t bilat_pop = 0

    if pop:
        for i in range(n_bins):
            if (g > (i - s0)) and (g < (i + s1)):
                bilat_pop += histo[i]
        out[0] = <dtype_t_out>bilat_pop
    else:
        out[0] = <dtype_t_out>0


cdef inline void _kernel_sum(dtype_t_out* out, Py_ssize_t odepth,
                             Py_ssize_t[::1] histo,
                             double pop, dtype_t g,
                             Py_ssize_t n_bins, Py_ssize_t mid_bin,
                             double p0, double p1,
                             Py_ssize_t s0, Py_ssize_t s1) nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t bilat_pop = 0
    cdef Py_ssize_t sum = 0

    if pop:
        for i in range(n_bins):
            if (g > (i - s0)) and (g < (i + s1)):
                bilat_pop += histo[i]
                sum += histo[i] * i
        if bilat_pop:
            out[0] = <dtype_t_out>sum
        else:
            out[0] = <dtype_t_out>0
    else:
        out[0] = <dtype_t_out>0


def _mean(dtype_t[:, ::1] image,
          char[:, ::1] footprint,
          char[:, ::1] mask,
          dtype_t_out[:, :, ::1] out,
          signed char shift_x, signed char shift_y, Py_ssize_t s0, Py_ssize_t s1,
          Py_ssize_t n_bins):

    _core(_kernel_mean[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, s0, s1, n_bins)


def _pop(dtype_t[:, ::1] image,
         char[:, ::1] footprint,
         char[:, ::1] mask,
         dtype_t_out[:, :, ::1] out,
         signed char shift_x, signed char shift_y, Py_ssize_t s0, Py_ssize_t s1,
         Py_ssize_t n_bins):

    _core(_kernel_pop[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, s0, s1, n_bins)


def _sum(dtype_t[:, ::1] image,
         char[:, ::1] footprint,
         char[:, ::1] mask,
         dtype_t_out[:, :, ::1] out,
         signed char shift_x, signed char shift_y, Py_ssize_t s0, Py_ssize_t s1,
         Py_ssize_t n_bins):

    _core(_kernel_sum[dtype_t_out, dtype_t], image, footprint, mask, out,
          shift_x, shift_y, 0, 0, s0, s1, n_bins)
