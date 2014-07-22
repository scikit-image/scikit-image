# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
cimport numpy as cnp
import numpy as np

def argmin2(cnp.float64_t[:] array):
    cdef cnp.float64_t min1 = np.inf
    cdef cnp.float64_t min2 = np.inf
    cdef Py_ssize_t i1 = 0
    cdef Py_ssize_t i2 = 0
    cdef Py_ssize_t i = 0

    while i < array.shape[0]:
        x = array[i]
        if x < min1 :
            min2 = min1
            i2 = i1
            min1 = x
            i1 = i
        elif x > min1 and x < min2 :
            min2 = x
            i2 = i
        i += 1
        
    return i2
