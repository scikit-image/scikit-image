"""template.py - Template matching
"""
import cython
cimport numpy as np
import numpy as np
import cv
from scipy.signal import fftconvolve

cdef extern from "math.h":
    double sqrt(double x)
    double fabs(double x)

@cython.boundscheck(False)
cdef void integral_image(float *image, double *ii, int width, int height):
    cdef double *prev_line = ii
    cdef double s
    cdef int x, y
    ii[0] = image[0]
    ii += 1
    image += 1
    for x in range(1, height):
        ii[0] = (image[0])
        ii[0] += (ii-1)[0]
        ii += 1
        image += 1
    for y in range(1, width):
        s = 0
        for x in range(0, height):
            s += image[0]
            ii[0] = s
            ii[0] += prev_line[0]
            ii += 1
            image += 1
            prev_line += 1

@cython.boundscheck(False)
cdef void integral_image2(float *image, double *ii2, int width, int height):
    cdef double *prev_line = ii2
    cdef double s
    cdef int x, y
    ii2[0] = (image[0])*(image[0])
    ii2 += 1
    image += 1
    for x in range(1, height):
        ii2[0] = (image[0]) * (image[0])
        ii2[0] += (ii2-1)[0]
        ii2 += 1
        image += 1
    for y in range(1, width):
        s = 0
        for x in range(0, height):
            s += (image[0]) * (image[0])
            ii2[0] = s
            ii2[0] += prev_line[0]
            ii2 += 1
            image += 1
            prev_line += 1

@cython.boundscheck(False)
def match_template(np.ndarray [float, ndim=2, mode="c"] image, np.ndarray[float, ndim=2, mode="c"] template):
    # convolve the image with template by frequency domain multiplication
    cdef np.ndarray result = np.ascontiguousarray(fftconvolve(image, template, mode="valid"))
#    out = np.empty((image.shape[0] - template.shape[0] + 1,image.shape[1] - template.shape[1] + 1), dtype=image.dtype)
#    cv.MatchTemplate(image, template, result, cv.CV_TM_CCORR)
    # calculate squared integral images used for normalization
    cdef np.ndarray integral_sqr = np.empty((image.shape[0], image.shape[1]))
    cdef np.ndarray integral_sqr2 = np.empty((image.shape[0], image.shape[1]))
    integral_image2(<float*>image.data, <double*>integral_sqr.data, image.shape[0], image.shape[1])
    # use inversed area for accuracy
    cdef double inv_area = 1.0 / (template.shape[0] * template.shape[1])
    # calculate template norm according to the following:
    # variance ** 2 = 1/K Sigma[(x_k - mean) ** 2] = 1/K Sigma[x_k ** 2] - mean ** 2
    cdef double template_norm = sqrt((np.std(template) ** 2 + np.mean(template) ** 2)) / sqrt(inv_area)
    cdef int sqstep = integral_sqr.strides[0] / sizeof(double)
    # define window of template size in squared integral image
    cdef double* q0 = <double*> integral_sqr.data
    cdef double* q1 = q0 + template.shape[1] 
    cdef double* q2 = <double*>integral_sqr.data + template.shape[0] * sqstep
    cdef double* q3 = q2 + template.shape[1]
    cdef int i, j, index
    cdef double num, window_sum2, normed
    # move window through convolution results, normalizing in the process
    for i in range(result.shape[0]):
        index = i * sqstep;
        for j in range(result.shape[1]):
            # calculate squared template window sum in the image
            window_sum2 = q0[index] - q1[index] - q2[index] + q3[index]
            normed = sqrt(window_sum2) * template_norm
            num = result[i, j]
            # enforce some limits
            if fabs(num) < normed:
                num /= normed
            elif fabs(num) < normed*1.125:
                if num > 0:
                    num = 1
                else:
                    num = -1                    
            else:
                num = 0
            result[i, j] = num
            index += 1
    return result
    

