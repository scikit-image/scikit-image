#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
import numpy as np
from libc.math cimport exp, fabs, sqrt
from libc.stdlib cimport malloc, free
from skimage._shared.interpolation cimport get_pixel2d, get_pixel3d


cdef inline double _gaussian_weight(double sigma, double value):
    return exp(- 0.5 * (value / sigma)**2)


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


def _denoise_bilateral2d(cnp.ndarray[dtype=cnp.double_t, ndim=2, mode='c'] image,
                         int win_size, double sigma_color,
                         double sigma_range, int bins, char mode,
                         double cval):

    cdef:
        Py_ssize_t rows = image.shape[0]
        Py_ssize_t cols = image.shape[1]
        Py_ssize_t window_ext = (win_size - 1) / 2
        double max_value = image.max()

        cnp.ndarray[dtype=cnp.double_t, ndim=2, mode='c'] out = \
        np.zeros((rows, cols), dtype=np.double)

        double* image_data = <double*>image.data
        double* out_data = <double*>out.data

        double* color_lut = _compute_color_lut(bins, sigma_color, max_value)
        double* range_lut = _compute_range_lut(win_size, sigma_range)

        Py_ssize_t r, c, wr, wc, kr, kc, rr, cc, pixel_addr
        double centre, value, weight, total_value, total_weight, \
               color_weight, range_weight, diff
        double dist_scale = bins / max_value

    for r in range(rows):
        for c in range(cols):
            pixel_addr = r * cols + c
            total_value = 0
            total_weight = 0
            centre = image_data[pixel_addr]
            for wr in range(- window_ext, window_ext + 1):
                rr = wr + r
                kr = wr + window_ext
                for wc in range(- window_ext, window_ext + 1):
                    cc = wc + c
                    kc = wc + window_ext

                    value = get_pixel2d(image_data, rows, cols,
                                        rr, cc, mode, cval)
                    diff = fabs(centre - value)

                    range_weight = range_lut[kr * win_size + kc]
                    color_weight = color_lut[<int>(diff * dist_scale)]

                    weight = range_weight * color_weight
                    total_value += value * weight
                    total_weight += weight

            out_data[pixel_addr] = total_value / total_weight

    free(color_lut)
    free(range_lut)

    return out


def _denoise_bilateral3d(cnp.ndarray[dtype=cnp.double_t, ndim=3, mode='c'] image,
                         int win_size, double sigma_color,
                         double sigma_range, int bins, char mode,
                         double cval):

    cdef:
        Py_ssize_t rows = image.shape[0]
        Py_ssize_t cols = image.shape[1]
        Py_ssize_t dims = image.shape[2]
        Py_ssize_t window_ext = (win_size - 1) / 2

        double max_value = image.max()

        cnp.ndarray[dtype=cnp.double_t, ndim=3, mode='c'] out = \
            np.zeros((rows, cols, dims), dtype=np.double)

        double* image_data = <double*>image.data
        double* out_data = <double*>out.data

        double* color_lut = _compute_color_lut(bins, sigma_color, max_value)
        double* range_lut = _compute_range_lut(win_size, sigma_range)

        Py_ssize_t r, c, d, wr, wc, kr, kc, rr, cc, pixel_addr
        double value, weight, dist, total_weight, color_weight, range_weight
        double dist_scale = bins / dims / max_value
        double* values = <double*>malloc(dims * sizeof(double))
        double* centres = <double*>malloc(dims * sizeof(double))
        double* total_values = <double*>malloc(dims * sizeof(double))

    for r in range(rows):
        for c in range(cols):
            pixel_addr = r * cols * dims + c * dims
            total_weight = 0
            for d in range(dims):
                total_values[d] = 0
                centres[d] = image_data[pixel_addr + d]
            for wr in range(- window_ext, window_ext + 1):
                rr = wr + r
                kr = wr + window_ext
                for wc in range(- window_ext, window_ext + 1):
                    cc = wc + c
                    kc = wc + window_ext

                    # save pixel values for all dims and compute euclidian
                    # distance between centre stack and current position
                    dist = 0
                    for d in range(dims):
                        value = get_pixel3d(image_data, rows, cols, dims,
                                            rr, cc, d, mode, cval)
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
