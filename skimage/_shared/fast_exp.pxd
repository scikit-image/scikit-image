#cython: initializedcheck=False
#cython: wraparound=False
#cython: boundscheck=False

cimport numpy as cnp
from .fused_numerics cimport np_floats

cdef extern from "fast_exp.h":
    double _fast_exp(double y) nogil
    float _fast_expf(float y) nogil


cdef inline np_floats _fast_exp_floats(np_floats x) nogil:
    if np_floats is cnp.float32_t:
        return _fast_expf(x)
    else:
        return _fast_exp(x)
