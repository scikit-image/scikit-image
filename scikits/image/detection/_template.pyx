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
        ii[0] += (ii - 1)[0]
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
        ii2[0] += (ii2 - 1)[0]
        ii2 += 1
        image += 1
    for y in range(1, width):
        s = 0
        for x in range(0, height):
            s += image[0]
            s += (image[0]) * (image[0])
            ii2[0] = s
            ii2[0] += prev_line[0]
            ii2 += 1
            image += 1
            prev_line += 1


@cython.boundscheck(False)
cdef void integral_images(float *image, double *ii, double *ii2, int width, int height):
    cdef double *prev_line = ii
    cdef double *prev_line2 = ii2
    cdef double s, s2
    cdef int x, y
    ii[0] = image[0]
    ii2[0] = (image[0] * image[0])
    ii += 1
    ii2 += 1
    image += 1
    for x in range(1, height):
        ii[0] = (image[0])
        ii[0] += (ii - 1)[0]
        ii += 1        
        ii2[0] = (image[0]) * (image[0])
        ii2[0] += (ii2 - 1)[0]
        ii2 += 1
        image += 1
    for y in range(1, width):
        s = 0
        s2 = 0
        for x in range(0, height):
            s += image[0]
            ii[0] = s
            ii[0] += prev_line[0]
            ii += 1
            s2 += (image[0]) * (image[0])
            ii2[0] = s2
            ii2[0] += prev_line2[0]
            ii2 += 1
            image += 1
            prev_line += 1
            prev_line2 += 1


@cython.boundscheck(False)
def match_template(np.ndarray[float, ndim=2, mode="c"] image,
        np.ndarray[float, ndim=2, mode="c"] template, int num_type):
    # convolve the image with template by frequency domain multiplication
    cdef np.ndarray[np.double_t, ndim=2] result
    result = np.ascontiguousarray(fftconvolve(image, template, mode="valid"), dtype=np.double)
    # calculate squared integral images used for normalization
    cdef np.ndarray integral_sum = np.zeros((image.shape[0], image.shape[1]))
    cdef np.ndarray integral_sqr = np.zeros((image.shape[0], image.shape[1]))
    if num_type == 1:
        integral_images(<float*>image.data, <double*>integral_sum.data, 
            <double*>integral_sqr.data, image.shape[0], image.shape[1])
    else:
        integral_image2(<float*>image.data, <double*>integral_sqr.data, image.shape[0], image.shape[1])
    # use inversed area for accuracy
    cdef double inv_area = 1.0 / (template.shape[0] * template.shape[1])
    # calculate template norm according to the following:
    # variance ** 2 = 1/K Sigma[(x_k - mean) ** 2] = 1/K Sigma[x_k ** 2] - mean ** 2
    cdef double template_norm
    cdef double template_mean = np.mean(template)
    
    if num_type == 0:
        template_norm = sqrt((np.std(template) ** 2 + template_mean ** 2)) / sqrt(inv_area)
    else:
        template_norm = sqrt((template_mean ** 2)) / sqrt(inv_area)
        
    cdef int sqstep = integral_sqr.strides[0] / sizeof(double)
    # define window of template size in squared integral image
    cdef double *p0, *p1, *p2, *p3
    cdef double *q0, *q1, *q2, *q3
    q0 = <double*> integral_sqr.data
    q1 = q0 + template.shape[1] 
    q2 = <double*>integral_sqr.data + template.shape[0] * sqstep
    q3 = q2 + template.shape[1]
    if num_type == 1:
        # define window of template size in summed integral image
        p0 = <double*> integral_sum.data
        p1 = p0 + template.shape[1] 
        p2 = <double*>integral_sum.data + template.shape[0] * sqstep
        p3 = p2 + template.shape[1]
    
    cdef int i, j, index
    cdef double num, window_sum2, window_mean2, normed, t, 
    # move window through convolution results, normalizing in the process
    for i in range(result.shape[0] - 1):
        index = i * sqstep;
        for j in range(result.shape[1] - 1):
            num = result[i, j]
            window_mean2 = 0
            if num_type == 1:
                t = p0[index] - p1[index] - p2[index] + p3[index]
                window_mean2 = t * t * inv_area
                num -= t*template_mean
        
            # calculate squared template window sum in the image
            window_sum2 = q0[index] - q1[index] - q2[index] + q3[index]
            normed = sqrt(window_sum2 - window_mean2) * template_norm
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
    # zero boundaries
    for i in range(result.shape[0]):
        result[i, -1] = 0
    for j in range(result.shape[1]):
        result[-1, j] = 0 
    return result
