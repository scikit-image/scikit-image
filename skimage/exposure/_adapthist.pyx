import numpy as np
cimport numpy as np

cdef extern from "clahe.h":
    ctypedef unsigned int kz_pixel_t
    int CLAHE(kz_pixel_t * pImage, unsigned int uiXRes, unsigned int uiYRes,
        kz_pixel_t Min,
        kz_pixel_t Max, unsigned int uiNrX, unsigned int uiNrY,
        unsigned int uiNrBins, float fCliplimit)


def _adapthist(np.ndarray[kz_pixel_t, ndim=2] image, min, max, nx, ny, nbins,
                 clip_limit=0):
    '''
    image -  to the input/output image
    min - Minimum greyvalue of input image (also becomes min of output image)
    max - Maximum greyvalue of input image (also becomes max of output image)
    nx - Number of tile regions in the X direction (min 2, max uiMAX_REG_X)
    ny - Number of tile regions in the Y direction (min 2, max uiMAX_REG_Y)
    nbins - Number of greybins for histogram ("dynamic range")
    clip_limit - Normalized cliplimit (higher values give more contrast)

    The number of "effective" greylevels in the output image is set by nbins;
    selecting a small value (eg. 128) speeds up processing and still produce
    an output image of good quality. The output image will have the same
    minimum and maximum value as the input image. A clip limit smaller than 1
    results in standard (non-contrast limited) AHE.
    '''
    # we need the image to be divisible by nx and ny
    uiYRes = image.shape[0] - image.shape[0] % ny
    uiXRes = image.shape[1] - image.shape[1] % nx

    cdef np.ndarray[kz_pixel_t, ndim = 1] flattened
    flattened = image[:uiYRes, :uiXRes].ravel()

    ret = CLAHE(< kz_pixel_t * > flattened.data, uiXRes, uiYRes, min,
                max, nx, ny, nbins, clip_limit * nbins)
    if ret < 0:
        print 'clahe error: ', ret
    return flattened.reshape((uiYRes, uiXRes))
