import numpy as np
cimport numpy as np

from skimage.transform import integral_image, integrate
from skimage import util


cdef inline int  _clip(np.int_t x, np.int_t low, np.int_t high):
    if(x > high):
        return high
    if(x < low):
        return low
    return x


cdef inline int _integ(np.int_t[:, :] img, np.int_t r1, np.int_t c1, np.int_t rl, np.int_t cl):

    r1 = _clip(r1, 0, img.shape[0] - 1)
    c1 = _clip(c1, 0, img.shape[1] - 1)

    r2 = _clip(r1 + rl, 0, img.shape[0] - 1)
    c2 = _clip(c1 + cl, 0, img.shape[1] - 1)

    cdef np.int_t r = img[r2, c2] + img[r1, c1] - img[r1, c2] - img[r2, c1]

    if (r < 0):
        return 0
    return r


def hessian_det_appx(np.ndarray[np.int_t, ndim=2] image, float sigma):

    cdef np.int_t[:, :] img = image
    cdef int size = int(3 * sigma)
    cdef np.ndarray[np.float_t, ndim = 2] out = np.zeros_like(img).astype(np.float)

    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef int r, c
    cdef int s2 = (size - 1) / 2
    cdef int s3 = size / 3
    cdef int l = size / 3
    cdef int w = size

    cdef float dxx, dyy, dxy

    if not size % 2:
        size += 1

    for r in range(height):
        for c in range(width):
        
            dxy =  _integ(img, r - s3, c + 1, s3, s3) + \
                   _integ(img, r + 1, c - s3, s3, s3) - \
                   _integ(img, r - s3, c - s3, s3, s3) - \
                   _integ(img, r + 1, c + 1, s3, s3)
            dxy = -dxy / w / w

            dxx = _integ(img, r - s3 + 1, c - s2, 2 * s3 - 1,w) - \
                  _integ(img, r - s3 + 1, c - s3 / 2, 2 * s3 - 1, s3) * 3
            dxx = -dxx / w / w

            dyy = _integ(img, r - s2, c - s2 + 1, w, 2 * s3 - 1) - \
                  _integ(img, r - s3 / 2, c - s3 + 1, s3, 2 * s3 - 1) * 3
            dyy = -dyy / w / w

            out[r, c] = (dxx * dyy - 0.81 * (dxy * dxy))

    return out
