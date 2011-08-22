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
cdef integral_image(np.ndarray[float, ndim=2, mode="c"] image):
    """
    Calculate the summed integral image.
    
    Parameters
    ----------
    image : array_like, dtype=float
        Source image.
        
    Returns
    -------
    output : ndarray, dtype=np.double_t
        Summed integral image.
    """
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] ii = np.zeros((image.shape[0], image.shape[1]))
    cdef double s
    cdef int x, y
    cdef int width, height
    height = image.shape[0]
    width = image.shape[1]
    ii[0, 0] = image[0, 0]

    for y in range(1, height):
        ii[y, 0] = image[y, 0] + ii[y - 1, 0]

    for x in range(1, width):
        s = 0
        for y in range(0, height):
            s += image[y, x]
            ii[y, x] = s + ii[y, x - 1]
    
    return ii


@cython.boundscheck(False)
cdef integral_image_sqr(np.ndarray[float, ndim=2, mode="c"] image):
    """
    Calculate the squared integral image.
    
    Parameters
    ----------
    image : array_like, dtype=float
        Source image.
        
    Returns
    -------
    output : ndarray, dtype=np.double_t
        Squared integral image.
    """
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] ii2 = np.zeros((image.shape[0], image.shape[1]))
    cdef double s
    cdef int x, y
    cdef int width, height
    height = image.shape[0]
    width = image.shape[1]
    ii2[0, 0] = image[0, 0] * image[0, 0]

    for y in range(1, height):
        ii2[y, 0] = image[y, 0] * image[y, 0] + ii2[y - 1, 0]

    for x in range(1, width):
        s = 0
        for y in range(0, height):
            s += image[y, x] * image[y, x]
            ii2[y, x] = s + ii2[y, x - 1]
    
    return ii2


@cython.boundscheck(False)
cdef integral_images(np.ndarray[float, ndim=2, mode="c"] image):
    """
    Calculate the summed and sqared integral image.
    
    Parameters
    ----------
    image : array_like, dtype=float
        Source image.
        
    Returns
    -------
    output : tuple (ndarray, ndarray) of type np.double_t
        Summed and squared integral image.
    """
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] ii = np.zeros((image.shape[0], image.shape[1]))
    cdef np.ndarray[np.double_t, ndim=2,  mode="c"] ii2 = np.zeros((image.shape[0], image.shape[1]))
    cdef double s, s2
    cdef int x, y
    cdef int width, height
    height = image.shape[0]
    width = image.shape[1]
    ii[0, 0] = image[0, 0]
    ii2[0, 0] = image[0, 0] * image[0, 0]

    for y in range(1, height):
        ii[y, 0] = image[y, 0] + ii[y - 1, 0]
        ii2[y, 0] = image[y, 0] * image[y, 0] + ii2[y - 1, 0]

    for x in range(1, width):
        s = 0
        s2 = 0
        for y in range(0, height):
            s += image[y, x]
            s2 += image[y, x] * image[y, x]
            ii[y, x] = s + ii[y, x - 1]
            ii2[y, x] = s2 + ii2[y, x - 1]
    
    return ii, ii2


@cython.boundscheck(False)
cdef double sum_integral(np.ndarray[np.double_t, ndim=2,  mode="c"] sat, 
        int r0, int c0, int r1, int c1):
    """
    Using a summed area table / integral image, calculate the sum
    over a given window.

    Parameters
    ----------
    sat : ndarray of double_t
        Summed area table / integral image.
    r0, c0 : int
        Top-left corner of block to be summed.
    r1, c1 : int
        Bottom-right corner of block to be summed.

    Returns
    -------
    S : int
        Sum over the given window.
    """
    cdef double S = 0

    S += sat[r1, c1]

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += sat[r0 - 1, c0 - 1]

    if (r0 - 1 >= 0):
        S -= sat[r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= sat[r1, c0 - 1]
    return S


@cython.boundscheck(False)
def match_template(np.ndarray[float, ndim=2, mode="c"] image,
        np.ndarray[float, ndim=2, mode="c"] template, int num_type):
    # convolve the image with template by frequency domain multiplication
    cdef np.ndarray[np.double_t, ndim=2] result
    result = np.ascontiguousarray(fftconvolve(image, np.fliplr(template), mode="valid"), dtype=np.double)
    # calculate squared integral images used for normalization
    cdef np.ndarray[np.double_t, ndim=2,  mode="c"] integral_sum
    cdef np.ndarray[np.double_t, ndim=2,  mode="c"] integral_sqr
    if num_type == 1:
        integral_sum, integral_sqr = integral_images(image)
    else:
        integral_sqr = integral_image_sqr(image)

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
        
    # define window of template size in squared integral image
    cdef int i, j
    cdef double num, window_sum2, window_mean2, normed, t, 
    # move window through convolution results, normalizing in the process
    for i in range(result.shape[0] - 1):
        for j in range(result.shape[1] - 1):
            num = result[i, j]
            window_mean2 = 0
            if num_type == 1:
                t = sum_integral(integral_sum, i, j, i + template.shape[0], j + template.shape[1])
                window_mean2 = t * t * inv_area
                num -= t*template_mean
        
            # calculate squared template window sum in the image
            window_sum2 = sum_integral(integral_sqr, i, j, i + template.shape[0], j + template.shape[1])
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
    # zero boundaries
    for i in range(result.shape[0]):
        result[i, -1] = 0
    for j in range(result.shape[1]):
        result[-1, j] = 0 
    return result
