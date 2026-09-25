# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: wraparound=False

"""Inner loop for the ridge-directed ring transform.

This implementation is based on the algorithm described by Eldad Afik in
Scientific Reports 5, 13584 (2015).
"""

from libc.math cimport M_PI, copysign, cos, fabs, sqrt

import numpy as np
cimport numpy as cnp


cdef double _COS_PI_8 = cos(M_PI / 8.0)


cdef inline Py_ssize_t _round(double value) noexcept nogil:
    if value >= 0:
        return <Py_ssize_t>(value + 0.5)
    return <Py_ssize_t>(value - 0.5)


cdef inline void _least_principal_direction(
    double hrr,
    double hrc,
    double hcc,
    double *cos_q,
    double *sin_q,
) noexcept nogil:
    cdef double d, tangent, denominator

    if hrc == 0:
        if hrr < hcc:
            cos_q[0] = 1.0
            sin_q[0] = 0.0
        else:
            cos_q[0] = 0.0
            sin_q[0] = 1.0
        return

    d = 0.5 * (hrr - hcc) / hrc
    if hrc > 0:
        tangent = -d - sqrt(d * d + 1.0)
    else:
        tangent = -d + sqrt(d * d + 1.0)
    denominator = sqrt(1.0 + tangent * tangent)
    cos_q[0] = 1.0 / denominator
    sin_q[0] = tangent / denominator


def _hough_ridge(
    const double[:, ::1] hrr,
    const double[:, ::1] hrc,
    const double[:, ::1] hcc,
    const double[:, ::1] curvature,
    double curvature_threshold,
    Py_ssize_t min_radius,
    Py_ssize_t max_radius,
):
    """Accumulate center votes from ridge pixels and their normal direction."""
    cdef:
        Py_ssize_t rows = curvature.shape[0]
        Py_ssize_t cols = curvature.shape[1]
        Py_ssize_t n_radii = max_radius - min_radius + 1
        Py_ssize_t row, col, radius, radius_index, center_row, center_col
        int sign
        double cos_q, sin_q, value
        cnp.float64_t[:, :, ::1] accumulator
        cnp.uint8_t[:, ::1] ridge_mask

    accumulator = np.zeros((n_radii, rows, cols), dtype=np.float64)
    ridge_mask = np.zeros((rows, cols), dtype=np.uint8)

    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            value = curvature[row, col]
            if value > curvature_threshold:
                continue

            _least_principal_direction(
                hrr[row, col], hrc[row, col], hcc[row, col], &cos_q, &sin_q
            )

            if fabs(cos_q) > _COS_PI_8:
                if value >= curvature[row - 1, col] or value >= curvature[row + 1, col]:
                    continue
            elif fabs(sin_q) > _COS_PI_8:
                if value >= curvature[row, col - 1] or value >= curvature[row, col + 1]:
                    continue
            elif copysign(1.0, sin_q) == copysign(1.0, cos_q):
                if value >= curvature[row - 1, col - 1] or value >= curvature[row + 1, col + 1]:
                    continue
            else:
                if value >= curvature[row - 1, col + 1] or value >= curvature[row + 1, col - 1]:
                    continue

            ridge_mask[row, col] = 1
            for radius_index in range(n_radii):
                radius = min_radius + radius_index
                for sign in (-1, 1):
                    center_row = row + sign * _round(cos_q * radius)
                    center_col = col + sign * _round(sin_q * radius)
                    if 0 <= center_row < rows and 0 <= center_col < cols:
                        accumulator[radius_index, center_row, center_col] += 1.0

    return np.asarray(accumulator), np.asarray(ridge_mask, dtype=bool)
