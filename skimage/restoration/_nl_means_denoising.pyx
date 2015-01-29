import numpy as np
from skimage import util
cimport numpy as np
cimport cython
from libc.math cimport exp

ctypedef np.float32_t DTYPE_t

cdef float DISTANCE_CUTOFF = 5.

@cython.boundscheck(False)
cdef inline float patch_distance_2d(DTYPE_t [:, :] p1,
                                    DTYPE_t [:, :] p2,
                                    DTYPE_t [:, ::] w, int s):
    """
    Compute a Gaussian distance between two image patches.

    Parameters
    ----------
    p1 : 2-D array_like
        First patch.
    p2 : 2-D array_like
        Second patch.
    w : 2-D array_like
        Array of weigths for the different pixels of the patches.
    s : int
        Linear size of the patches.

    Returns
    -------
    distance : float
        Gaussian distance between the two patches

    Notes
    -----
    The returned distance is given by

    .. math::  \exp( -w (p1 - p2)^2)
    """
    cdef int i, j
    cdef int center = s / 2
    # Check if central pixel is too different in the 2 patches
    cdef float tmp_diff = p1[center, center] - p2[center, center]
    cdef float init = w[center, center] * tmp_diff * tmp_diff
    if init > 1:
        return 0.
    cdef float distance = 0
    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > DISTANCE_CUTOFF:
            return 0.
        for j in range(s):
            tmp_diff = p1[i, j] - p2[i, j]
            distance += (w[i, j] * tmp_diff * tmp_diff)
    distance = exp(-distance)
    return distance


@cython.boundscheck(False)
cdef inline float patch_distance_2drgb(DTYPE_t [:, :, :] p1,
                                       DTYPE_t [:, :, :] p2,
                                       DTYPE_t [:, ::] w, int s):
    """
    Compute a Gaussian distance between two image patches.

    Parameters
    ----------
    p1 : 3-D array_like
        First patch, 2D image with last dimension corresponding to channels.
    p2 : 3-D array_like
        Second patch, 2D image with last dimension corresponding to channels.
    w : 2-D array_like
        Array of weights for the different pixels of the patches.
    s : int
        Linear size of the patches.

    Returns
    -------
    distance : float
        Gaussian distance between the two patches

    Notes
    -----
    The returned distance is given by

    .. math::  \exp( -w (p1 - p2)^2)
    """
    cdef int i, j
    cdef int center = s / 2
    cdef int color
    cdef float tmp_diff = 0
    cdef float distance = 0
    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > DISTANCE_CUTOFF:
            return 0.
        for j in range(s):
            for color in range(3):
                tmp_diff = p1[i, j, color] - p2[i, j, color]
                distance += w[i, j] * tmp_diff * tmp_diff
    distance = exp(-distance)
    return distance


@cython.boundscheck(False)
cdef inline float patch_distance_3d(DTYPE_t [:, :, :] p1,
                                    DTYPE_t [:, :, :] p2,
                                    DTYPE_t [:, :, ::] w, int s):
    """
    Compute a Gaussian distance between two image patches.

    Parameters
    ----------
    p1 : 3-D array_like
        First patch.
    p2 : 3-D array_like
        Second patch.
    w : 3-D array_like
        Array of weights for the different pixels of the patches.
    s : int
        Linear size of the patches.

    Returns
    -------
    distance : float
        Gaussian distance between the two patches

    Notes
    -----
    The returned distance is given by

    .. math::  \exp( -w (p1 - p2)^2)
    """
    cdef int i, j, k
    cdef float distance = 0
    cdef float tmp_diff
    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > DISTANCE_CUTOFF:
            return 0.
        for j in range(s):
            for k in range(s):
                tmp_diff = p1[i, j, k] - p2[i, j, k]
                distance += w[i, j, k] * tmp_diff * tmp_diff
    distance = exp(-distance)
    return distance


@cython.cdivision(True)
@cython.boundscheck(False)
def _nl_means_denoising_2d(image, int s=7, int d=13, float h=0.1):
    """
    Perform non-local means denoising on 2-D RGB image

    Parameters
    ----------
    image : ndarray
        Input RGB image to be denoised
    s : int, optional
        Size of patches used for denoising
    d : int, optional
        Maximal distance in pixels where to search patches used for denoising
    h : float, optional
        Cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.

    Returns
    -------
    result : ndarray
        Denoised image, of same shape as input image.
    """
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef int n_row, n_col, n_ch
    n_row, n_col, n_ch = image.shape
    cdef int offset = s / 2
    cdef int x_row, x_col, i, j, color
    cdef int x_row_start, x_row_end, x_col_start, x_col_end
    cdef int x_row_start_i, x_row_end_i, x_col_start_j, x_col_end_j
    cdef DTYPE_t [::1] new_values = np.zeros(n_ch).astype(np.float32)
    cdef DTYPE_t [:, :, ::1] padded = np.ascontiguousarray(util.pad(image,
                       ((offset, offset), (offset, offset), (0, 0)),
                        mode='reflect').astype(np.float32))
    cdef DTYPE_t [:, :, ::1] result = padded.copy()
    cdef float A = ((s - 1.) / 4.)
    cdef float new_value
    cdef float weight_sum, weight
    xg_row, xg_col = np.mgrid[-offset:offset + 1, -offset:offset + 1]
    cdef DTYPE_t [:, ::1] w = np.ascontiguousarray(np.exp(
                             -(xg_row ** 2 + xg_col ** 2) / (2 * A ** 2)).
                             astype(np.float32))
    cdef float distance
    w = 1. / (n_ch * np.sum(w) * h ** 2) * w
    # Coordinates of central pixel and patch bounds
    for x_row in range(offset, n_row + offset):
        x_row_start = x_row - offset
        x_row_end = x_row + offset + 1
        for x_col in range(offset, n_col + offset):
            for color in range(n_ch):
                new_values[color] = 0
            weight_sum = 0
            x_col_start = x_col - offset
            x_col_end = x_col + offset + 1
            # Coordinates of test pixel and patch bounds
            for i in range(max(- d, offset - x_row),
                           min(d + 1, n_row + offset - x_row)):
                x_row_start_i = x_row_start + i
                x_row_end_i = x_row_end + i
                for j in range(max(- d, offset - x_col),
                               min(d + 1, n_col + offset - x_col)):
                    x_col_start_j = x_col_start + j
                    x_col_end_j = x_col_end + j
                    if n_ch == 1:
                        weight = patch_distance_2d(
                                 padded[x_row_start: x_row_end,
                                        x_col_start: x_col_end, 0],
                                 padded[x_row_start_i: x_row_end_i,
                                        x_col_start_j: x_col_end_j, 0],
                                 w, s)
                    else:
                        weight = patch_distance_2drgb(
                                 padded[x_row_start: x_row_end,
                                        x_col_start: x_col_end, :],
                                 padded[x_row_start_i: x_row_end_i,
                                        x_col_start_j: x_col_end_j, :],
                                        w, s)
                    weight_sum += weight
                    for color in range(n_ch):
                        new_values[color] += weight * padded[x_row + i,
                                                             x_col + j, color]
            for color in range(n_ch):
                result[x_row, x_col, color] = new_values[color] / weight_sum
    return result[offset:-offset, offset:-offset]


@cython.cdivision(True)
@cython.boundscheck(False)
def _nl_means_denoising_3d(image, int s=7,
            int d=13, float h=0.1):
    """
    Perform non-local means denoising on 3-D array

    Parameters
    ----------
    image : ndarray
        Input data to be denoised.
    s : int, optional
        Size of patches used for denoising.
    d : int, optional
        Maximal distance in pixels where to search patches used for denoising.
    h : float, optional
        Cut-off distance (in gray levels).

    Returns
    -------
    result : ndarray
        Denoised image, of same shape as input image.
    """
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef int n_pln, n_row, n_col
    n_pln, n_row, n_col = image.shape
    cdef int offset = s / 2
    # padd the image so that boundaries are denoised as well
    cdef DTYPE_t [:, :, ::1] padded = np.ascontiguousarray(util.pad(
                                        image.astype(np.float32),
                                        offset, mode='reflect'))
    cdef DTYPE_t [:, :, ::1] result = padded.copy()
    cdef float A = ((s - 1.) / 4.)
    cdef float new_value
    cdef float weight_sum, weight
    xg_pln, xg_row, xg_col = np.mgrid[-offset: offset + 1,
                                      -offset: offset + 1,
                                      -offset: offset + 1]
    cdef DTYPE_t [:, :, ::1] w = np.ascontiguousarray(np.exp(
                            -(xg_pln ** 2 + xg_row ** 2 + xg_col ** 2) /
                            (2 * A ** 2)).astype(np.float32))
    cdef float distance
    cdef int x_pln, x_row, x_col, i, j, k
    cdef int x_pln_start, x_pln_end, x_row_start, x_row_end, \
             x_col_start, x_col_end
    cdef int x_pln_start_i, x_pln_end_i, x_row_start_j, x_row_end_j, \
             x_col_start_k, x_col_end_k
    w = 1. / (np.sum(w) * h ** 2) * w
    # Coordinates of central pixel and patch bounds
    for x_pln in range(offset, n_pln + offset):
        x_pln_start = x_pln - offset
        x_pln_end = x_pln + offset + 1
        for x_row in range(offset, n_row + offset):
            x_row_start = x_row - offset
            x_row_end = x_row + offset + 1
            for x_col in range(offset, n_col + offset):
                x_col_start = x_col - offset
                x_col_end = x_col + offset + 1
                new_value = 0
                weight_sum = 0
                # Coordinates of test pixel and patch bounds
                for i in range(max(- d, offset - x_pln),
                               min(d + 1, n_pln + offset - x_pln)):
                    x_pln_start_i = x_pln_start + i
                    x_pln_end_i = x_pln_end + i
                    for j in range(max(- d, offset - x_row),
                                   min(d + 1, n_row + offset - x_row)):
                        x_row_start_j = x_row_start + j
                        x_row_end_j = x_row_end + j
                        for k in range(max(- d, offset - x_col),
                                   min(d + 1, n_col + offset - x_col)):
                            x_col_start_k = x_col_start + k
                            x_col_end_k = x_col_end + k
                            weight = patch_distance_3d(
                                    padded[x_pln_start: x_pln_end,
                                           x_row_start: x_row_end,
                                           x_col_start: x_col_end],
                                    padded[x_pln_start_i: x_pln_end_i,
                                           x_row_start_j: x_row_end_j,
                                           x_col_start_k: x_col_end_k],
                                    w, s)
                            weight_sum += weight
                            new_value += weight * padded[x_pln + i,
                                                         x_row + j, x_col + k]
                result[x_pln, x_row, x_col] = new_value / weight_sum
    return result[offset:-offset, offset:-offset, offset:-offset]

#-------------- Accelerated algorithm of Froment 2015 ------------------

@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline float _integral_to_distance_2d(DTYPE_t [:, ::] integral,
                        int x_row, int x_col, int offset, float h2s2):
    """
    References
    ----------
    Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
    Denoising. Image Processing On Line, 2014, vol. 4, p. 300-326.

    Used in _fast_nl_means_denoising_2d
    """
    cdef float distance
    distance =  integral[x_row + offset, x_col + offset] + \
                integral[x_row - offset, x_col - offset] - \
                integral[x_row - offset, x_col + offset] - \
                integral[x_row + offset, x_col - offset]
    distance /= h2s2
    return distance

@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline float _integral_to_distance_3d(DTYPE_t [:, :, ::] integral,
                    int x_pln, int x_row, int x_col, int offset,
                    float s_cube_h_square):
    """
    References
    ----------
    Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
    Denoising. Image Processing On Line, 2014, vol. 4, p. 300-326.

    Used in _fast_nl_means_denoising_3d
    """

    cdef float distance
    distance = (integral[x_pln + offset, x_row + offset, x_col + offset]
              - integral[x_pln - offset, x_row - offset, x_col - offset]
              + integral[x_pln - offset, x_row - offset, x_col + offset]
              + integral[x_pln - offset, x_row + offset, x_col - offset]
              + integral[x_pln + offset, x_row - offset, x_col - offset]
              - integral[x_pln - offset, x_row + offset, x_col + offset]
              - integral[x_pln + offset, x_row - offset, x_col + offset]
              - integral[x_pln + offset, x_row + offset, x_col - offset])
    distance /= s_cube_h_square
    return distance


@cython.cdivision(True)
@cython.boundscheck(False)
def _fast_nl_means_denoising_2d(image, int s=7, int d=13, float h=0.1):
    """
    Perform fast non-local means denoising on 2-D array, with the outer
    loop on patch shifts in order to reduce the number of operations.

    Parameters
    ----------
    image : ndarray
        2-D input data to be denoised, grayscale or RGB.
    s : int, optional
        Size of patches used for denoising.
    d : int, optional
        Maximal distance in pixels where to search patches used for denoising.
    h : float, optional
        Cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.

    Returns
    -------
    result : ndarray
        Denoised image, of same shape as input image.
    """
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef int offset = s / 2
    # Image padding: we need to account for patch size, possible shift,
    # + 1 for the boundary effects in finite differences
    cdef int pad_size = offset + d + 1
    cdef DTYPE_t [:, :, ::1] padded = np.ascontiguousarray(util.pad(image,
                          ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                          mode='reflect').astype(np.float32))
    cdef DTYPE_t [:, :, ::1] result = np.zeros_like(padded)
    cdef DTYPE_t [:, ::1] weights = np.zeros_like(padded[..., 0], order='C')
    cdef DTYPE_t [:, ::1] integral = np.zeros_like(padded[..., 0], order='C')
    cdef int n_row, n_col, n_ch, t_row, t_col, x_row, x_col
    cdef float weight, distance
    cdef float alpha
    cdef float h2 = h ** 2.
    cdef float s2 = s ** 2.
    n_row, n_col, n_ch = image.shape
    cdef float h2s2 = n_ch * h2 * s2
    n_row += 2 * pad_size
    n_col += 2 * pad_size
    # Outer loops on patch shifts
    # With t2 >= 0, reference patch is always on the left of test patch
    for t_row in range(-d, d + 1):
        for t_col in range(0, d + 1):
            # alpha is to account for patches on the same column
            # distance is computed twice in this case
            if t_col == 0 and t_row is not 0:
                alpha = 0.5
            else:
                alpha = 1.
            integral = np.zeros_like(padded[..., 0], order='C')
            for x_row in range(max(1, - t_row), min(n_row, n_row - t_row)):
                for x_col in range(max(1, - t_col), min(n_col, n_col - t_col)):
                    if n_ch == 1:
                        distance = (padded[x_row, x_col, 0] -
                                    padded[x_row + t_row, x_col + t_col, 0])**2
                    else:
                        distance = ((padded[x_row, x_col, 0] -
                                    padded[x_row + t_row, x_col + t_col, 0])**2
                                    +(padded[x_row, x_col, 1] -
                                    padded[x_row + t_row, x_col + t_col, 1])**2
                                    +(padded[x_row, x_col, 2] -
                                    padded[x_row + t_row, x_col + t_col, 2])**2)
                    integral[x_row, x_col] = distance + \
                                     integral[x_row - 1, x_col] + \
                                     integral[x_row, x_col - 1] \
                                      - integral[x_row - 1, x_col - 1]
            for x_row in range(max(offset, offset - t_row),
                           min(n_row - offset, n_row - offset - t_row)):
                for x_col in range(max(offset, offset - t_col),
                               min(n_col - offset, n_col - offset - t_col)):
                    distance = _integral_to_distance_2d(integral, x_row, x_col,
                                                     offset, h2s2)
                    # exp of large negative numbers is close to zero
                    if distance > DISTANCE_CUTOFF:
                        continue
                    weight = alpha * exp(-distance)
                    weights[x_row, x_col] += weight
                    weights[x_row + t_row, x_col + t_col] += weight
                    for ch in range(n_ch):
                        result[x_row, x_col, ch] += weight * \
                                        padded[x_row + t_row, x_col + t_col, ch]
                        result[x_row + t_row, x_col + t_col, ch] += \
                                        weight * padded[x_row, x_col, ch]
    # Normalize pixel values using sum of weights of contributing patches
    for x_row in range(offset, n_row - offset):
        for x_col in range(offset, n_col - offset):
            for channel in range(n_ch):
                # No risk of division by zero, since the contribution
                # of a null shift is strictly positive
                result[x_row, x_col, channel] /= weights[x_row, x_col]
    return result[pad_size: - pad_size, pad_size: - pad_size]


@cython.cdivision(True)
@cython.boundscheck(False)
def _fast_nl_means_denoising_3d(image, int s=5, int d=7, float h=0.1):
    """
    Perform fast non-local means denoising on 3-D array, with the outer
    loop on patch shifts in order to reduce the number of operations.

    Parameters
    ----------
    image : ndarray
        3-D input data to be denoised.
    s : int, optional
        Size of patches used for denoising.
    d : int, optional
        Maximal distance in pixels where to search patches used for denoising.
    h : float, optional
        cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.

    Returns
    -------
    result : ndarray
        Denoised image, of same shape as input image.
    """
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef int offset = s / 2
    # Image padding: we need to account for patch size, possible shift,
    # + 1 for the boundary effects in finite differences
    cdef int pad_size = offset + d + 1
    cdef DTYPE_t [:, :, ::1] padded = np.ascontiguousarray(util.pad(image,
                                pad_size, mode='reflect').astype(np.float32))
    cdef DTYPE_t [:, :, ::1] result = np.zeros_like(padded)
    cdef DTYPE_t [:, :, ::1] weights = np.zeros_like(padded)
    cdef DTYPE_t [:, :, ::1] integral = np.zeros_like(padded)
    cdef int n_pln, n_row, n_col, t_pln, t_row, t_col, \
             x_pln, x_row, x_col
    cdef int x_pln_integral_min, x_pln_integral_max, \
             x_row_integral_min, x_row_integral_max, \
             x_col_integral_min, x_col_integral_max
    cdef int x_pln_dist_min, x_pln_dist_max, x_row_dist_min, x_row_dist_max, \
             x_col_dist_min, x_col_dist_max
    cdef float weight, distance
    cdef float alpha
    cdef float h_square = h ** 2.
    cdef float s_cube = s ** 3.
    cdef float s_cube_h_square = h_square * s_cube
    n_pln, n_row, n_col = image.shape
    n_pln += 2 * pad_size
    n_row += 2 * pad_size
    n_col += 2 * pad_size
    # Outer loops on patch shifts
    # With t2 >= 0, reference patch is always on the left of test patch
    for t_pln in range(-d, d + 1):
        x_pln_integral_min = max(1, - t_pln)
        x_pln_integral_max = min(n_pln, n_pln - t_pln)
        x_pln_dist_min = max(offset, offset - t_pln)
        x_pln_dist_max = min(n_pln - offset, n_pln - offset - t_pln)
        for t_row in range(-d, d + 1):
            x_row_integral_min = max(1, - t_row)
            x_row_integral_max = min(n_row, n_row - t_row)
            x_row_dist_min = max(offset, offset - t_row)
            x_row_dist_max = min(n_row - offset, n_row - offset - t_row)
            for t_col in range(0, d + 1):
                x_col_integral_min = max(1, - t_col)
                x_col_integral_max = min(n_col, n_col - t_col)
                x_col_dist_min = max(offset, offset - t_col)
                x_col_dist_max = min(n_col - offset, n_col - offset - t_col)
                # alpha is to account for patches on the same column
                # distance is computed twice in this case
                if t_col == 0 and (t_pln is not 0 or t_row is not 0):
                    alpha = 0.5
                else:
                    alpha = 1.
                integral = np.zeros_like(padded)
                for x_pln in range(x_pln_integral_min, x_pln_integral_max):
                    for x_row in range(x_row_integral_min, x_row_integral_max):
                        for x_col in range(x_col_integral_min,
                                           x_col_integral_max):
                            integral[x_pln, x_row, x_col] = \
                                ((padded[x_pln, x_row, x_col] -
                                padded[x_pln + t_pln, x_row + t_row,
                                       x_col + t_col])**2 +
                                integral[x_pln - 1, x_row, x_col] +
                                integral[x_pln, x_row - 1, x_col] +
                                integral[x_pln, x_row, x_col - 1] +
                                integral[x_pln - 1, x_row - 1, x_col - 1]
                                - integral[x_pln - 1, x_row - 1, x_col]
                                - integral[x_pln, x_row - 1, x_col - 1]
                                - integral[x_pln - 1, x_row, x_col - 1])
                for x_pln in range(x_pln_dist_min, x_pln_dist_max):
                    for x_row in range(x_row_dist_min, x_row_dist_max):
                        for x_col in range(x_col_dist_min, x_col_dist_max):
                            distance = _integral_to_distance_3d(integral,
                                        x_pln, x_row, x_col, offset,
                                        s_cube_h_square)
                            # exp of large negative numbers is close to zero
                            if distance > DISTANCE_CUTOFF:
                                continue
                            weight = alpha * exp(-distance)
                            weights[x_pln, x_row, x_col] += weight
                            weights[x_pln + t_pln, x_row + t_row,
                                                   x_col + t_col] += weight
                            result[x_pln, x_row, x_col] += weight * \
                                    padded[x_pln + t_pln, x_row + t_row,
                                                          x_col + t_col]
                            result[x_pln + t_pln, x_row + t_row,
                                                  x_col + t_col] += weight * \
                                                  padded[x_pln, x_row, x_col]
    for x_pln in range(offset, n_pln - offset):
        for x_row in range(offset, n_row - offset):
            for x_col in range(offset, n_col - offset):
                # I think there is no risk of division by zero
                # except in padded zone
                result[x_pln, x_row, x_col] /= weights[x_pln, x_row, x_col]
    return result[pad_size: - pad_size, pad_size: - pad_size,
                                        pad_size: -pad_size]
