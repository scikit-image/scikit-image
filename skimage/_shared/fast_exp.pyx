from .fused_numerics cimport np_floats
from .fast_exp cimport _fast_exp


def fast_exp(np_floats x):
    return _fast_exp(x)
