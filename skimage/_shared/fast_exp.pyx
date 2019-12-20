import numpy as np
cimport numpy as cnp
from .fused_numerics cimport np_floats

cdef extern from "fast_exp.h":
    double fast_exp(double y) nogil


def fast_exp(np_floats x):
    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64
    cdef double x_f64=x
    return dtype.type(fast_exp(x_f64))
