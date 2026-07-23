#cython: initializedcheck=False
#cython: wraparound=False
#cython: boundscheck=False

cimport numpy as cnp
from .fused_numerics cimport np_floats

cdef extern from "fast_exp.h":
    cnp.float64_t _fast_exp(cnp.float64_t y) noexcept nogil
    cnp.float32_t _fast_expf(cnp.float32_t y) noexcept nogil


cdef inline np_floats _fast_exp_floats(np_floats x) noexcept nogil:
    if np_floats is cnp.float32_t:
        return _fast_expf(x)
    else:
        return _fast_exp(x)
