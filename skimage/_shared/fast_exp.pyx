from .fused_numerics cimport np_floats
from .fast_exp cimport exp_func


def exp_func_py(np_floats x):
    return exp_func(x)
