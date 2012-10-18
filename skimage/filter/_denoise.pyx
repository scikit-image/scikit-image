#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
import numpy as np
from libc.math cimport exp, fabs, sqrt
from libc.stdlib cimport malloc, free
from skimage._shared.interpolation cimport get_pixel3d
from skimage.util import img_as_float


cdef inline double _gaussian_weight(double sigma, double value):
    return exp(-0.5 * (value / sigma)**2)


cdef double* _compute_color_lut(int bins, double sigma, double max_value):

    cdef:
        double* color_lut = <double*>malloc(bins * sizeof(double))
        Py_ssize_t b

    for b in range(bins):
        color_lut[b] = _gaussian_weight(sigma, b * max_value / bins)

    return color_lut


cdef double* _compute_range_lut(int win_size, double sigma):

    cdef:
        double* range_lut = <double*>malloc(win_size**2 * sizeof(double))
        Py_ssize_t kr, kc
        Py_ssize_t window_ext = (win_size - 1) / 2
        double dist

    for kr in range(win_size):
        for kc in range(win_size):
            dist = sqrt((kr - window_ext)**2 + (kc - window_ext)**2)
            range_lut[kr * win_size + kc] = _gaussian_weight(sigma, dist)

    return range_lut


def denoise_bilateral(image, int win_size=5, sigma_range=None,
                      double sigma_spatial=1, int bins=10000, mode='constant',
                      double cval=0):
    """Denoise image using bilateral filter.

    This is an edge-preserving and noise reducing denoising filter. It averages
    pixels based on their spatial closeness and radiometric similarity.

    Spatial closeness is measured by the gaussian function of the euclidian
    distance between two pixels and a certain standard deviation
    (`sigma_spatial`).

    Radiometric similarity is measured by the gaussian function of the euclidian
    distance between two color values and a certain standard deviation
    (`sigma_range`).

    Parameters
    ----------
    image : ndarray
        Input image.
    win_size : int
        Window size for filtering.
    sigma_range : float
        Standard deviation for grayvalue/color distance (radiometric
        similarity). A larger value results in averaging of pixels with larger
        radiometric differences. Note, that the image will be converted using
        the `img_as_float` function and thus the standard deviation is in
        respect to the range `[0, 1]`.
    sigma_spatial : float
        Standard deviation for range distance. A larger value results in
        averaging of pixels with larger spatial differences.
    bins : int
        Number of discrete values for gaussian weights of color filtering.
        A larger value results in improved accuracy.
    mode : string
        How to handle values outside the image borders. See
        `scipy.ndimage.map_coordinates` for detail.
    cval : string
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    denoised : ndarray
        Denoised image.

    References
    ----------
    .. [1] http://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf

    """

    image = np.atleast_3d(img_as_float(image))

    cdef:
        Py_ssize_t rows = image.shape[0]
        Py_ssize_t cols = image.shape[1]
        Py_ssize_t dims = image.shape[2]
        Py_ssize_t window_ext = (win_size - 1) / 2

        double max_value = image.max()

        cnp.ndarray[dtype=cnp.double_t, ndim=3, mode='c'] cimage = \
            np.ascontiguousarray(image)
        cnp.ndarray[dtype=cnp.double_t, ndim=3, mode='c'] out = \
            np.zeros((rows, cols, dims), dtype=np.double)

        double* image_data = <double*>cimage.data
        double* out_data = <double*>out.data

        double* color_lut = _compute_color_lut(bins, sigma_range, max_value)
        double* range_lut = _compute_range_lut(win_size, sigma_spatial)

        Py_ssize_t r, c, d, wr, wc, kr, kc, rr, cc, pixel_addr
        double value, weight, dist, total_weight, csigma_range, color_weight, \
               range_weight
        double dist_scale = bins / dims / max_value
        double* values = <double*>malloc(dims * sizeof(double))
        double* centres = <double*>malloc(dims * sizeof(double))
        double* total_values = <double*>malloc(dims * sizeof(double))

    if sigma_range is None:
        csigma_range = image.std()
    else:
        csigma_range = sigma_range

    if mode not in ('constant', 'wrap', 'reflect', 'nearest'):
        raise ValueError("Invalid mode specified.  Please use "
                         "`constant`, `nearest`, `wrap` or `reflect`.")
    cdef char cmode = ord(mode[0].upper())

    for r in range(rows):
        for c in range(cols):
            pixel_addr = r * cols * dims + c * dims
            total_weight = 0
            for d in range(dims):
                total_values[d] = 0
                centres[d] = image_data[pixel_addr + d]
            for wr in range(-window_ext, window_ext + 1):
                rr = wr + r
                kr = wr + window_ext
                for wc in range(-window_ext, window_ext + 1):
                    cc = wc + c
                    kc = wc + window_ext

                    # save pixel values for all dims and compute euclidian
                    # distance between centre stack and current position
                    dist = 0
                    for d in range(dims):
                        value = get_pixel3d(image_data, rows, cols, dims,
                                            rr, cc, d, cmode, cval)
                        values[d] = value
                        dist += (centres[d] - value)**2
                    dist = sqrt(dist)

                    range_weight = range_lut[kr * win_size + kc]
                    color_weight = color_lut[<int>(dist * dist_scale)]

                    weight = range_weight * color_weight
                    for d in range(dims):
                        total_values[d] += values[d] * weight
                    total_weight += weight
            for d in range(dims):
                out_data[pixel_addr + d] = total_values[d] / total_weight

    free(color_lut)
    free(range_lut)
    free(values)
    free(centres)
    free(total_values)

    return out
