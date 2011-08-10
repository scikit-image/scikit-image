"""template.py - Template matching
"""
import cython
cimport numpy as np
import numpy as np
import cv
from scipy.signal import fftconvolve

#def extnot2(np.ndarray[np.uint8_t, ndim=2, mode="c"] image not None):
@cython.boundscheck(False)
def match_template(np.ndarray image, np.ndarray template):
    size = image.shape
    template_size = template.shape
    out = np.empty((size[0] - template_size[0] + 1,size[1] - template_size[0] + 1), dtype=np.float32)
    result = np.ascontiguousarray(fftconvolve(image, template, mode="valid"))
    integral = np.empty((result.shape[0]+1, result.shape[1]+1))
    integral_sqr = np.empty((result.shape[0]+1, result.shape[1]+1))

    cv.Integral(result, integral, integral_sqr)
    
    area = (template.shape[0] * template.shape[1])
    template_norm = sqrt((np.sdv(template) ** 2 + np.mean(template) ** 2) * area)
    
    cdef double *q0, *q1, *q2, *q3
    cdef q0 = integral_sqr.data
    cdef q1 = q0 + integral_sqr.shape[1]
    cdef q2 = (double*)(integral_sqr.data + integral_sqr.shape[0]*integral_sqr.strides[0])
    cdef q3 = q2 + integral_sqr.shape[1]

    cdef int sqstep = integral_sqr.data ? (int)(integral_sqr.strides[0] / sizeof(double)) : 0;
    
    int i, j, k;
    for i in range(result.shape[0]):
        int idx = i * sumstep;
        int idx2 = i * sqstep;
        for j in range(result.shape[1]):
            cdef double num = template[i, j]
            window_sum2 = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2]
            t = sqrt(window_sum2)*template_norm
            if fabs(num) < t:
                num /= t;
            elif fabs(num) < t*1.125:
                num = num > 0 ? 1 : -1;
            else:
                num = 0;
            result[i, j] = (float)num
    return result
    

