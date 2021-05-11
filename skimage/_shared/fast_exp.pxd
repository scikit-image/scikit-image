#cython: initializedcheck=False
#cython: wraparound=False
#cython: boundscheck=False

cimport numpy as cnp
from .fused_numerics cimport np_floats

cdef extern from "fast_exp.h":
    double fast_exp(double y) nogil
    float fast_expf(float y) nogil


cdef inline np_floats _fast_exp(np_floats x) nogil:
    if np_floats is cnp.float32_t:
        return fast_expf(x)
    else:
        return fast_exp(x)
