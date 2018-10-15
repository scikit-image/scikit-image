#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
import numpy as np
from libc.math cimport exp, fabs, sqrt
from libc.float cimport DBL_MAX
from .._shared.interpolation cimport get_pixel3d
from .._shared.fused_numerics cimport np_floats


def _gaussian_weight(np_floats sigma_sqr, np_floats value):
    return exp(-0.5 * value * value / sigma_sqr)


cdef np_floats[:] _compute_range_lut(Py_ssize_t win_size, np_floats sigma,
                                     np_floats[::1] range_lut):

    cdef:
        Py_ssize_t kr, kc, dr, dc
        Py_ssize_t window_ext = (win_size - 1) / 2
        np_floats dist

    sigma *= sigma

    for kr in range(win_size):
        for kc in range(win_size):
            dr = kr - window_ext
            dc = kc - window_ext
            dist = sqrt(dr * dr + dc * dc)
            range_lut[kr * win_size + kc] = _gaussian_weight(sigma, dist)

    return range_lut


cdef inline Py_ssize_t Py_ssize_t_min(Py_ssize_t value1, Py_ssize_t value2):
    if value1 < value2:
        return value1
    else:
        return value2


def _denoise_bilateral(np_floats[:, :, ::1] image, np_floats max_value,
                       Py_ssize_t win_size, np_floats sigma_color,
                       np_floats sigma_spatial, Py_ssize_t bins, mode,
                       np_floats cval, np_floats[::1] color_lut,
                       np_floats[::1] range_lut, np_floats[::1] empty_dims,
                       np_floats[:, :, ::1] out):
    cdef:
        Py_ssize_t rows = image.shape[0]
        Py_ssize_t cols = image.shape[1]
        Py_ssize_t dims = image.shape[2]
        Py_ssize_t window_ext = (win_size - 1) / 2
        Py_ssize_t max_color_lut_bin = bins - 1

        Py_ssize_t r, c, d, wr, wc, kr, kc, rr, cc, pixel_addr, color_lut_bin
        np_floats value, weight, dist, total_weight, csigma_color, color_weight, \
               range_weight, t
        np_floats dist_scale
        np_floats[:] values
        np_floats[:] centres
        np_floats[:] total_values

    if sigma_color is None:
        csigma_color = image.std()
    else:
        csigma_color = sigma_color

    if mode not in ('constant', 'wrap', 'symmetric', 'reflect', 'edge'):
        raise ValueError("Invalid mode specified.  Please use `constant`, "
                         "`edge`, `wrap`, `symmetric` or `reflect`.")
    cdef char cmode = ord(mode[0].upper())


    dist_scale = bins / dims / max_value
    values = empty_dims.copy()
    centres = empty_dims.copy()
    total_values = empty_dims.copy()

    for r in range(rows):
        for c in range(cols):
            total_weight = 0
            for d in range(dims):
                total_values[d] = 0
                centres[d] = image[r, c, d]
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
                        value = get_pixel3d(&image[0, 0, 0], rows, cols, dims,
                                            rr, cc, d, cmode, cval)
                        values[d] = value
                        t = centres[d] - value
                        dist += t * t
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


def _denoise_tv_bregman(np_floats[:, :, ::1] image, np_floats weight,
                        int max_iter, np_floats eps,
                        char isotropic, np_floats[:, :, ::1] out):
    cdef:
        Py_ssize_t rows = image.shape[0]
        Py_ssize_t cols = image.shape[1]
        Py_ssize_t dims = image.shape[2]
        Py_ssize_t rows2 = rows + 2
        Py_ssize_t cols2 = cols + 2
        Py_ssize_t r, c, k

        Py_ssize_t total = rows * cols * dims

    shape_ext = (rows2, cols2, dims)
    u = out.copy()

    cdef:
        np_floats[:, :, ::1] cu = u

        np_floats[:, :, ::1] dx = out.copy()
        np_floats[:, :, ::1] dy = out.copy()
        np_floats[:, :, ::1] bx = out.copy()
        np_floats[:, :, ::1] by = out.copy()

        np_floats ux, uy, uprev, unew, bxx, byy, dxx, dyy, s, tx, ty
        int i = 0
        np_floats lam = 2 * weight
        np_floats rmse = DBL_MAX
        np_floats norm = (weight + 4 * lam)

    height, width = image.shape[:2]
    u_height, u_width = u.shape[:2]
    u[1:u_height-1, 1:u_width-1] = image

    # reflect image
    u[0, 1:u_width-1] = image[1, :]
    u[1:u_height-1, 0] = image[:, 1]
    u[u_height-1, 1:u_width-1] = image[height-2, :]
    u[1:u_height-1, u_width-1] = image[:, width-2]

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
                        ) + weight * image[r - 1, c - 1, k]
                    ) / norm
                    cu[r, c, k] = unew

                    # update root mean square error
                    tx = unew - uprev
                    rmse += tx * tx

                    bxx = bx[r, c, k]
                    byy = by[r, c, k]

                    # d_subproblem after reference [4]
                    if isotropic:
                        tx = ux + bxx
                        ty = uy + byy
                        s = sqrt(tx * tx + ty * ty)
                        dxx = s * lam * tx / (s * lam + 1)
                        dyy = s * lam * ty / (s * lam + 1)

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

    return np.squeeze(np.asarray(u[1:u_height-1, 1:u_width-1]))
