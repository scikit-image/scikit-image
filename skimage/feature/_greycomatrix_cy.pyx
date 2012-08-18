"""Cython implementation for computing a grey level co-occurance matrix
"""

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double sin(double)
    double cos(double)

@cython.boundscheck(False)
def _glcm_loop(np.ndarray[dtype=np.uint8_t, ndim=2, 
                          negative_indices=False, mode='c'] image,
               np.ndarray[dtype=np.float64_t, ndim=1,
                          negative_indices=False, mode='c'] distances,
               np.ndarray[dtype=np.float64_t, ndim=1,
                          negative_indices=False, mode='c'] angles,
               int levels,
               np.ndarray[dtype=np.uint32_t, ndim=4, 
                          negative_indices=False, mode='c'] out
               ):
    """Perform co-occurnace matrix accumulation
    
    Parameters
    ----------
    image : ndarray
        Input image, which is converted to the uint8 data type.
    distances : ndarray
        List of pixel pair distance offsets.
    angles : ndarray
        List of pixel pair angles in radians.
    levels : int
        The input image should contain integers in [0, levels-1],
        where levels indicate the number of grey-levels counted
        (typically 256 for an 8-bit image)
    out : ndarray
        On input a 4D array of zeros, and on output it contains 
        the results of the GLCM computation. 

    """
    cdef:
        np.int32_t a_inx, d_idx
        np.int32_t r, c, rows, cols, row, col
        np.int32_t i, j
        
    rows = image.shape[0]
    cols = image.shape[1]
     
    for a_idx, angle in enumerate(angles):
        for d_idx, distance in enumerate(distances):
            for r in range(rows):
                for c in range(cols):
                    i = image[r, c]

                    # compute the location of the offset pixel
                    row = r + <int>(sin(angle) * distance + 0.5)
                    col = c + <int>(cos(angle) * distance + 0.5);
                    
                    # make sure the offset is within bounds
                    if row >= 0 and row < rows and \
                       col >= 0 and col < cols:
                        j = image[row, col]
                        
                        if i >= 0 and i < levels and \
                           j >= 0 and j < levels:
                            out[i, j, d_idx, a_idx] += 1
