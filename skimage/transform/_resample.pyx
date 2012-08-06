# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

cimport numpy as cnp

from cython.parallel import prange


def _resample_float32(
    cnp.ndarray[cnp.float32_t, ndim=3] arr_in,
    cnp.ndarray[cnp.float32_t, ndim=3] arr_out):

    cdef Py_ssize_t inh = arr_in.shape[0]
    cdef Py_ssize_t inw = arr_in.shape[1]
    cdef Py_ssize_t ind = arr_in.shape[2]
    cdef Py_ssize_t outh = arr_out.shape[0]
    cdef Py_ssize_t outw = arr_out.shape[1]
    cdef Py_ssize_t outd = arr_out.shape[2]

    cdef Py_ssize_t iy, ix, iz
    cdef Py_ssize_t oy, ox, oz

    for oy in prange(outh, nogil=True):
        if inh != outh:
            iy = <Py_ssize_t>(oy * (1. * inh / outh))
        else:
            iy = oy
        for ox in range(outw):
            if inw != outw:
                ix = <Py_ssize_t>(1. * ox * (1. *  inw / outw))
            else:
                ix = ox
            for oz in range(outd):
                if ind != outd:
                    iz = <Py_ssize_t>(1. * oz * (1. * ind / outd))
                else:
                    iz = oz
                arr_out[oy, ox, oz] = arr_in[iy, ix, iz]
