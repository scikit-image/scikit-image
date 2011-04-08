cimport cython

import numpy as np
cimport numpy as np

np.import_array()


cdef extern from "math.h":
    double sqrt(double)
    double ceil(double)
    double round(double)


cdef double PI_2 = 1.5707963267948966
cdef double NEG_PI_2 = -PI_2


@cython.boundscheck(False)
def _hough(np.ndarray img, np.ndarray[ndim=1, dtype=np.double_t] theta=None):
    
    if img.ndim != 2:
        raise ValueError('The input image must be 2D.')

    # Compute the array of angles and their sine and cosine
    cdef np.ndarray[ndim=1, dtype=np.double_t] ctheta
    cdef np.ndarray[ndim=1, dtype=np.double_t] stheta

    if theta is None:
        theta = np.linspace(PI_2, NEG_PI_2, 180) 

    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    # compute the bins and allocate the output array
    cdef np.ndarray[ndim=2, dtype=np.uint64_t] out
    cdef np.ndarray[ndim=1, dtype=np.double_t] bins
    cdef int max_distance, offset 

    max_distance = 2 * <int>ceil((sqrt(img.shape[0] * img.shape[0] + 
                                       img.shape[1] * img.shape[1])))
    out = np.zeros((max_distance, theta.shape[0]), dtype=np.uint64)
    bins = np.linspace(-max_distance / 2.0, max_distance / 2.0, max_distance)
    offset = max_distance / 2
    
    # compute the nonzero indexes
    cdef np.ndarray[ndim=1, dtype=np.int32_t] x_idxs, y_idxs
    y_idxs, x_idxs = np.PyArray_Nonzero(img)

    # finally, run the transform
    cdef int nidxs, nthetas, i, j, x, y, out_idx
    nidxs = y_idxs.shape[0] # x and y are the same shape
    nthetas = theta.shape[0]
    for i in range(nidxs):
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(nthetas):
            out_idx = <int>round((ctheta[j] * x + stheta[j] * y)) + offset
            out[out_idx, j] += 1

    return out, theta, bins


