import cython
cimport numpy as np

cdef extern from "convolve.h": 
    void convolve(float* src, float* dst, float* kernel, int width, int height, \
        int kernel_width, int kernel_height, int anchor_x, int anchor_y)
    
@cython.boundscheck(False)
@cython.wraparound(False)
def pyconvolve(np.ndarray[float, ndim=2, mode="c"] image, np.ndarray[float, ndim=2, mode="c"] dest, \
        np.ndarray[float, ndim=2, mode="c"] kernel, anchor=(-1, -1)):
    assert anchor[0] >= -1 and anchor[0] < kernel.shape[1]
    assert anchor[1] >= -1 and anchor[1] < kernel.shape[0]
    convolve(<float*> image.data, <float*> dest.data, <float*> kernel.data, image.shape[1], image.shape[0], kernel.shape[1], kernel.shape[0], anchor[0], anchor[1])
