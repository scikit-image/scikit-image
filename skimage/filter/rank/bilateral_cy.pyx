#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport log

from .core_cy cimport uint8_t, uint16_t, dtype_t, _core


cdef inline dtype_t _kernel_mean(Py_ssize_t* histo, float pop,
                                 dtype_t g,
                                 Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                 float p0, float p1,
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
            return <dtype_t>(mean / bilat_pop)
        else:
            return <dtype_t>(0)
    else:
        return <dtype_t>(0)


cdef inline dtype_t _kernel_pop(Py_ssize_t* histo, float pop,
                                dtype_t g,
                                Py_ssize_t max_bin, Py_ssize_t mid_bin,
                                float p0, float p1,
                                Py_ssize_t s0, Py_ssize_t s1):

    cdef Py_ssize_t i
    cdef Py_ssize_t bilat_pop = 0

    if pop:
        for i in range(max_bin):
            if (g > (i - s0)) and (g < (i + s1)):
                bilat_pop += histo[i]
        return <dtype_t>(bilat_pop)
    else:
        return <dtype_t>(0)


def _mean(dtype_t[:, ::1] image,
          char[:, ::1] selem,
          char[:, ::1] mask,
          dtype_t[:, ::1] out,
          char shift_x, char shift_y, Py_ssize_t s0, Py_ssize_t s1,
          Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_mean[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, s0, s1, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_mean[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, s0, s1, max_bin)


def _pop(dtype_t[:, ::1] image,
        char[:, ::1] selem,
        char[:, ::1] mask,
        dtype_t[:, ::1] out,
        char shift_x, char shift_y, Py_ssize_t s0, Py_ssize_t s1,
        Py_ssize_t max_bin):

    if dtype_t is uint8_t:
        _core[uint8_t](_kernel_pop[uint8_t], image, selem, mask, out,
                       shift_x, shift_y, 0, 0, s0, s1, max_bin)
    elif dtype_t is uint16_t:
        _core[uint16_t](_kernel_pop[uint16_t], image, selem, mask, out,
                        shift_x, shift_y, 0, 0, s0, s1, max_bin)
