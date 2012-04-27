import numpy as np
import scipy.ndimage
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def _threshold_adaptive(np.ndarray[np.double_t, ndim=2] image, int block_size,
                        method, double offset, mode, param):
    cdef int r, c
    cdef np.ndarray[np.double_t, ndim=2] thresh_image

    if method == 'generic':
        thresh_image = scipy.ndimage.generic_filter(image, param, block_size,
            mode=mode)
    elif method == 'gaussian':
        if param is None:
            # automatically determine sigma which covers > 99% of distribution
            sigma = (block_size - 1) / 6.0
        thresh_image = scipy.ndimage.gaussian_filter(image, sigma, mode=mode)
    elif method == 'mean':
        mask = 1. / block_size * np.ones((block_size,))
        # separation of filters to speedup convolution
        thresh_image = scipy.ndimage.convolve1d(image, mask, axis=0, mode=mode)
        thresh_image = scipy.ndimage.convolve1d(thresh_image, mask, axis=1,
            mode=mode)
    elif method == 'median':
        thresh_image = scipy.ndimage.median_filter(image, block_size, mode=mode)

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            thresh_image[r,c] = image[r,c] > (thresh_image[r,c] - offset)

    return thresh_image.astype('bool')
