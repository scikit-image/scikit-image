#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
import numpy as np
from libc.math cimport exp, fabs, sqrt
from libc.float cimport DBL_MAX
from .._shared.interpolation cimport get_pixel3d
from ..util import img_as_float


cdef inline double _gaussian_weight(double sigma, double value):
    return exp(-0.5 * (value / sigma)**2)


cdef double[:] _compute_color_lut(Py_ssize_t bins, double sigma, double max_value):

    cdef:
        double[:] color_lut = np.empty(bins, dtype=np.double)
        Py_ssize_t b

    for b in range(bins):
        color_lut[b] = _gaussian_weight(sigma, b * max_value / bins)

    return color_lut


cdef double[:] _compute_range_lut(Py_ssize_t win_size, double sigma):

    cdef:
        double[:] range_lut = np.empty(win_size**2, dtype=np.double)
        Py_ssize_t kr, kc
        Py_ssize_t window_ext = (win_size - 1) / 2
        double dist

    for kr in range(win_size):
        for kc in range(win_size):
            dist = sqrt((kr - window_ext)**2 + (kc - window_ext)**2)
            range_lut[kr * win_size + kc] = _gaussian_weight(sigma, dist)

    return range_lut


cdef inline Py_ssize_t Py_ssize_t_min(Py_ssize_t value1, Py_ssize_t value2):
    if value1 < value2:
        return value1
    else:
        return value2


def _denoise_bilateral(image, Py_ssize_t win_size, sigma_range,
                      double sigma_spatial, Py_ssize_t bins,
                      mode, double cval):
    image = np.atleast_3d(img_as_float(image))

    # if image.max() is 0, then dist_scale can have an unverified value
    # and color_lut[<int>(dist * dist_scale)] may cause a segmentation fault
    # so we verify we have a positive image and that the max is not 0.0.
    if image.min() < 0.0:
        raise ValueError("Image must contain only positive values")

    cdef:
        Py_ssize_t rows = image.shape[0]
        Py_ssize_t cols = image.shape[1]
        Py_ssize_t dims = image.shape[2]
        Py_ssize_t window_ext = (win_size - 1) / 2
        Py_ssize_t max_color_lut_bin = bins - 1

        double max_value

        double[:, :, ::1] cimage
        double[:, :, ::1] out

        double[:] color_lut
        double[:] range_lut

        Py_ssize_t r, c, d, wr, wc, kr, kc, rr, cc, pixel_addr, color_lut_bin
        double value, weight, dist, total_weight, csigma_range, color_weight, \
               range_weight
        double dist_scale
        double[:] values
        double[:] centres
        double[:] total_values

    if sigma_range is None:
        csigma_range = image.std()
    else:
        csigma_range = sigma_range

    max_value = image.max()

    if max_value == 0.0:
        raise ValueError("The maximum value found in the image was 0.")

    if mode not in ('constant', 'wrap', 'symmetric', 'reflect', 'edge'):
        raise ValueError("Invalid mode specified.  Please use `constant`, "
                         "`edge`, `wrap`, `symmetric` or `reflect`.")
    cdef char cmode = ord(mode[0].upper())

    cimage = np.ascontiguousarray(image)

    out = np.zeros((rows, cols, dims), dtype=np.double)
    color_lut = _compute_color_lut(bins, csigma_range, max_value)
    range_lut = _compute_range_lut(win_size, sigma_spatial)
    dist_scale = bins / dims / max_value
    values = np.empty(dims, dtype=np.double)
    centres = np.empty(dims, dtype=np.double)
    total_values = np.empty(dims, dtype=np.double)

    for r in range(rows):
        for c in range(cols):
            total_weight = 0
            for d in range(dims):
                total_values[d] = 0
                centres[d] = cimage[r, c, d]
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
                        value = get_pixel3d(&cimage[0, 0, 0], rows, cols, dims,
                                            rr, cc, d, cmode, cval)
                        values[d] = value
                        dist += (centres[d] - value)**2
                    dist = sqrt(dist)

                    range_weight = range_lut[kr * win_size + kc]

                    color_lut_bin = Py_ssize_t_min(
                        <Py_ssize_t>(dist * dist_scale), max_color_lut_bin)
                    color_weight = color_lut[color_lut_bin]

                    weight = range_weight * color_weight
                    for d in range(dims):
                        total_values[d] += values[d] * weight
                    total_weight += weight
            for d in range(dims):
                out[r, c, d] = total_values[d] / total_weight

    return np.squeeze(np.asarray(out))


def _denoise_tv_bregman(image, double weight, int max_iter, double eps,
                       char isotropic):
    image = np.atleast_3d(img_as_float(image))

    cdef:
        Py_ssize_t rows = image.shape[0]
        Py_ssize_t cols = image.shape[1]
        Py_ssize_t dims = image.shape[2]
        Py_ssize_t rows2 = rows + 2
        Py_ssize_t cols2 = cols + 2
        Py_ssize_t r, c, k

        Py_ssize_t total = rows * cols * dims

    shape_ext = (rows2, cols2, dims)
    u = np.zeros(shape_ext, dtype=np.double)

    cdef:
        double[:, :, ::1] cimage = np.ascontiguousarray(image)
        double[:, :, ::1] cu = u

        double[:, :, ::1] dx = np.zeros(shape_ext, dtype=np.double)
        double[:, :, ::1] dy = np.zeros(shape_ext, dtype=np.double)
        double[:, :, ::1] bx = np.zeros(shape_ext, dtype=np.double)
        double[:, :, ::1] by = np.zeros(shape_ext, dtype=np.double)

        double ux, uy, uprev, unew, bxx, byy, dxx, dyy, s
        int i = 0
        double lam = 2 * weight
        double rmse = DBL_MAX
        double norm = (weight + 4 * lam)

    u[1:-1, 1:-1] = image

    # reflect image
    u[0, 1:-1] = image[1, :]
    u[1:-1, 0] = image[:, 1]
    u[-1, 1:-1] = image[-2, :]
    u[1:-1, -1] = image[:, -2]

    while i < max_iter and rmse > eps:

        rmse = 0

        for k in range(dims):
            for r in range(1, rows + 1):
                for c in range(1, cols + 1):

                    uprev = cu[r, c, k]

                    # forward derivatives
                    ux = cu[r, c + 1, k] - uprev
                    uy = cu[r + 1, c, k] - uprev

                    # Gauss-Seidel method
                    unew = (
                        lam * (
                            + cu[r + 1, c, k]
                            + cu[r - 1, c, k]
                            + cu[r, c + 1, k]
                            + cu[r, c - 1, k]

                            + dx[r, c - 1, k]
                            - dx[r, c, k]
                            + dy[r - 1, c, k]
                            - dy[r, c, k]

                            - bx[r, c - 1, k]
                            + bx[r, c, k]
                            - by[r - 1, c, k]
                            + by[r, c, k]
                        ) + weight * cimage[r - 1, c - 1, k]
                    ) / norm
                    cu[r, c, k] = unew

                    # update root mean square error
                    rmse += (unew - uprev)**2

                    bxx = bx[r, c, k]
                    byy = by[r, c, k]

                    # d_subproblem after reference [4]
                    if isotropic:
                        s = sqrt((ux + bxx)**2 + (uy + byy)**2)
                        dxx = s * lam * (ux + bxx) / (s * lam + 1)
                        dyy = s * lam * (uy + byy) / (s * lam + 1)

                    else:
                        s = ux + bxx
                        if s > 1 / lam:
                            dxx = s - 1/lam
                        elif s < -1 / lam:
                            dxx = s + 1 / lam
                        else:
                            dxx = 0
                        s = uy + byy
                        if s > 1 / lam:
                            dyy = s - 1 / lam
                        elif s < -1 / lam:
                            dyy = s + 1 / lam
                        else:
                            dyy = 0

                    dx[r, c, k] = dxx
                    dy[r, c, k] = dyy

                    bx[r, c, k] += ux - dxx
                    by[r, c, k] += uy - dyy

        rmse = sqrt(rmse / total)
        i += 1

    return np.squeeze(np.asarray(u[1:-1, 1:-1]))
