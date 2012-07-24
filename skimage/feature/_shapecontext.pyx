# -*- python -*-
#cython: cdivision=True

import cython
import numpy as np
cimport numpy as np

"""

"""

cdef extern from "math.h":
    float sqrt(float x)
    float atan2(float y, float x)

@cython.boundscheck(False)
def _shapecontext(np.ndarray[np.float64_t, ndim=2, mode="c"] image, 
                  float r_min, float r_max,
                  int current_pixel_x, int current_pixel_y, 
                  int radial_bins=5, int polar_bins=12):

    cdef int cols = image.shape[0]
    cdef int rows = image.shape[1]

    cdef int x, y, x_diff, y_diff, r_idx, theta_idx, tmp
    cdef float r, theta 

    if r_min <= 0:
        r_min = 1
    cdef np.ndarray[np.float64_t, ndim=1] r_array = \
        np.logspace(np.log10(r_min), np.log10(r_max), radial_bins + 1, base=10)
    cdef np.ndarray[np.float64_t, ndim=1] theta_array = \
        np.linspace(-np.pi, np.pi, polar_bins + 1)

    cdef np.ndarray[np.float64_t, ndim=2] bin_histogram = \
        np.zeros((radial_bins, polar_bins), dtype=float)

    cdef int r_max_int = int(r_max)

    #print r_array
    #print theta_array

    for x in xrange(max(current_pixel_x - r_max_int, 0), 
                    min(current_pixel_x + r_max_int, cols)):
        for y in xrange(max(current_pixel_y - r_max_int, 0), 
                        min(current_pixel_y + r_max_int, rows)):
            x_diff = current_pixel_x - x
            y_diff = current_pixel_y - y
            #print "x_diff = ", x_diff
            #print "y_diff = ", y_diff
                        
            r = sqrt(x_diff*x_diff + y_diff*y_diff)
            #print "r = ", r 
            theta = atan2(y_diff, x_diff)
            #print "theta = ", theta
            
            r_idx = -1
            for tmp in xrange(radial_bins):
                if r > r_array[tmp] and r < r_array[tmp + 1]:
                    r_idx = tmp
            
            theta_idx = -1
            for tmp in xrange(polar_bins):
                if theta > theta_array[tmp] and theta < theta_array[tmp + 1]:
                    theta_idx = tmp
            
            #print "r_idx = ", r_idx, "theta_idx = ", theta_idx
            
            if r_idx != -1 and theta_idx != -1 and image[x, y] != 0:
                bin_histogram[r_idx, theta_idx] += 1

    return bin_histogram
