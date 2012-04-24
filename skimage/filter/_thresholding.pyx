import numpy as np
import scipy.ndimage
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def _threshold_adaptive(np.ndarray[np.double_t, ndim=2] image,
                        int block_size, double offset, method):
    cdef int r, c
    cdef np.ndarray[np.float64_t, ndim=2] mean_image
    if method == 'gaussian':
        # covers > 99% of distribution
        sigma = (block_size - 1) / 6.0
        mean_image = scipy.ndimage.gaussian_filter(image, sigma)
    elif method == 'mean':
        mask = 1. / block_size**2 * np.ones((block_size, block_size))
        mean_image = scipy.ndimage.convolve(image, mask)
    elif method == 'median':
        mean_image = scipy.ndimage.median_filter(image, block_size)

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            mean_image[r,c] = image[r,c] > (mean_image[r,c] - offset)

    return mean_image.astype('bool')
