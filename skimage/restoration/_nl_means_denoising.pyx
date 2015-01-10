import numpy as np
from skimage import util
cimport numpy as np
cimport cython
from libc.math cimport exp

ctypedef np.float32_t DTYPE_t

cdef eps = 1.e-8


@cython.boundscheck(False)
cdef inline float patch_distance_2d(DTYPE_t [:, :] p1,
                                    DTYPE_t [:, :] p2,
                                    DTYPE_t [:, ::] w, int s):
    cdef int i, j
    cdef int center = s / 2
    # Check if central pixel is too different in the 2 patches
    cdef float tmp_diff = p1[center, center] - p2[center, center]
    cdef float init = w[center, center] * tmp_diff * tmp_diff
    if init > 1:
        return eps
    cdef float distance = 0
    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > 4:
            return eps
        for j in range(s):
            tmp_diff = p1[i, j] - p2[i, j]
            distance += w[i, j] * tmp_diff * tmp_diff
    distance = exp(- distance)
    return distance


@cython.boundscheck(False)
cdef inline float patch_distance_2drgb(DTYPE_t [:, :, :] p1,
                                       DTYPE_t [:, :, :] p2,
                                       DTYPE_t [:, ::] w, int s):
    cdef int i, j
    cdef int center = s / 2
    cdef int color
    cdef float tmp_diff = 0
    cdef float distance = 0
    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > 4:
            return eps
        for j in range(s):
            for color in range(3):
                tmp_diff = p1[i, j, color] - p2[i, j, color]
                distance += w[i, j] * tmp_diff * tmp_diff
    distance = exp(- distance)
    return distance


@cython.boundscheck(False)
cdef inline float patch_distance_3d(DTYPE_t [:, :, :] p1,
                                    DTYPE_t [:, :, :] p2,
                                    DTYPE_t [:, :, ::] w, int s):
    cdef int i, j, k
    cdef float distance = 0
    cdef float tmp_diff
    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > 4:
            return eps
        for j in range(s):
            for k in range(s):
                tmp_diff = p1[i, j, k] - p2[i, j, k]
                distance += w[i, j, k] * tmp_diff * tmp_diff
    distance = exp(- distance)
    return distance


@cython.cdivision(True)
@cython.boundscheck(False)
def _nl_means_denoising_2d(image, int s=7, int d=13, float h=0.1):
    """
    Perform non-local means denoising on 2-D array

    Parameters
    ----------
    image: ndarray
        input data to be denoised

    s: int, optional
        size of patches used for denoising

    d: int, optional
        maximal distance in pixels where to search patches used for denoising

    h: float, optional
        cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.
    """
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef int n_x, n_y
    n_x, n_y = image.shape
    cdef int offset = s / 2
    # padd the image so that boundaries are denoised as well
    cdef DTYPE_t [:, ::1] padded = np.ascontiguousarray(util.pad(image,
                                offset, mode='reflect').astype(np.float32))
    cdef DTYPE_t [:, ::1] result = padded.copy()
    cdef float A = ((s - 1.) / 4.)
    cdef float new_value
    cdef float weight_sum, weight
    xg, yg = np.mgrid[-offset:offset + 1, -offset:offset + 1]
    cdef DTYPE_t [:, ::1] w = np.ascontiguousarray(np.exp(
                                    - (xg ** 2 + yg ** 2) / (2 * A ** 2)).
                                    astype(np.float32))
    cdef float distance
    cdef int x, y, i, j
    cdef int x_start, x_end, y_start, y_end
    cdef int x_start_i, x_end_i, y_start_j, y_end_j
    w = 1. / (np.sum(w) * h ** 2.) * w
    # Coordinates of central pixel and patch bounds
    for x in range(offset, n_x + offset):
        x_start = x - offset
        x_end = x + offset + 1
        for y in range(offset, n_y + offset):
            new_value = 0
            weight_sum = 0
            y_start = y - offset
            y_end = y + offset + 1
            # Coordinates of test pixel and patch bounds
            for i in range(max(- d, offset - x),
                           min(d + 1, n_x - x - 1)):
                x_start_i = x_start + i
                x_end_i = x_end + i
                for j in range(max(- d, offset - y),
                               min(d + 1, n_y - y - 1)):
                    y_start_j = y_start + j
                    y_end_j = y_end + j
                    weight = patch_distance_2d(
                                padded[x_start: x_end,
                                       y_start: y_end],
                                padded[x_start_i: x_end_i,
                                       y_start_j: y_end_j],
                                w, s)
                    weight_sum += weight
                    new_value += weight * padded[x + i, y + j]
            result[x, y] = new_value / weight_sum
    return result[offset:-offset, offset:-offset]


@cython.cdivision(True)
@cython.boundscheck(False)
def _nl_means_denoising_2drgb(image, int s=7, int d=13, float h=0.1):
    """
    Perform non-local means denoising on 2-D RGB image

    Parameters
    ----------
    image: ndarray
        input RGB image to be denoised

    s: int, optional
        size of patches used for denoising

    d: int, optional
        maximal distance in pixels where to search patches used for denoising

    h: float, optional
        cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.
    """
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef int n_x, n_y
    n_x, n_y, _ = image.shape
    cdef int offset = s / 2
    cdef int x, y, i, j, color
    cdef int x_start, x_end, y_start, y_end
    cdef int x_start_i, x_end_i, y_start_j, y_end_j
    cdef DTYPE_t [::1] new_values = np.zeros(3).astype(np.float32)
    cdef DTYPE_t [:, :, ::1] padded = np.ascontiguousarray(util.pad(image,
                       ((offset, offset), (offset, offset), (0, 0)),
                        mode='reflect').astype(np.float32))
    cdef DTYPE_t [:, :, ::1] result = padded.copy()
    cdef float A = ((s - 1.) / 4.)
    cdef float new_value
    cdef float weight_sum, weight
    xg, yg = np.mgrid[-offset:offset + 1, -offset:offset + 1]
    cdef DTYPE_t [:, ::1] w = np.ascontiguousarray(np.exp(
                                    - (xg ** 2 + yg ** 2) / (2 * A ** 2)).
                                    astype(np.float32))
    cdef float distance
    w = 1. / (np.sum(w) * h ** 2) * w
    # Coordinates of central pixel and patch bounds
    for x in range(offset, n_x + offset):
        x_start = x - offset
        x_end = x + offset + 1
        for y in range(offset, n_y + offset):
            for color in range(3):
                new_values[color] = 0
            weight_sum = 0
            y_start = y - offset
            y_end = y + offset + 1
            # Coordinates of test pixel and patch bounds
            for i in range(max(- d, offset - x),
                           min(d + 1, n_x - x - 1)):
                x_start_i = x_start + i
                x_end_i = x_end + i
                for j in range(max(- d, offset - y),
                               min(d + 1, n_y - y - 1)):
                    y_start_j = y_start + j
                    y_end_j = y_end + j
                    weight = patch_distance_2drgb(
                                padded[x_start: x_end,
                                       y_start: y_end, :],
                                padded[x_start_i: x_end_i,
                                       y_start_j: y_end_j, :],
                                w, s)
                    weight_sum += weight
                    for color in range(3):
                        new_values[color] += weight * padded[x + i, y + j,
                                                             color]
            for color in range(3):
                result[x, y, color] = new_values[color] / weight_sum
    return result[offset:-offset, offset:-offset]


@cython.cdivision(True)
@cython.boundscheck(False)
def _nl_means_denoising_3d(image, int s=7,
            int d=13, float h=0.1):
    """
    Perform non-local means denoising on 3-D array

    Parameters
    ----------
    image: ndarray
        input data to be denoised

    s: int, optional
        size of patches used for denoising

    d: int, optional
        maximal distance in pixels where to search patches used for denoising

    h: float, optional
        cut-off distance (in gray levels)
    """
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef int n_x, n_y, n_z
    n_x, n_y, n_z = image.shape
    cdef int offset = s / 2
    # padd the image so that boundaries are denoised as well
    cdef DTYPE_t [:, :, ::1] padded = np.ascontiguousarray(util.pad(
                                        image.astype(np.float32),
                                        offset, mode='reflect'))
    cdef DTYPE_t [:, :, ::1] result = padded.copy()
    cdef float A = ((s - 1.) / 4.)
    cdef float new_value
    cdef float weight_sum, weight
    xg, yg, zg = np.mgrid[-offset: offset + 1, -offset: offset + 1,
                            -offset: offset + 1]
    cdef DTYPE_t [:, :, ::1] w = np.ascontiguousarray(np.exp(
                            - (xg ** 2 + yg ** 2 + zg ** 2) / (2 * A ** 2)).
                                astype(np.float32))
    cdef float distance
    cdef int x, y, z, i, j, k
    cdef int x_start, x_end, y_start, y_end, z_start, z_end
    cdef int x_start_i, x_end_i, y_start_j, y_end_j, z_start_k, z_end_k
    w = 1. / (np.sum(w) * h ** 2) * w
    # Coordinates of central pixel and patch bounds
    for x in range(offset, n_x + offset):
        x_start = x - offset
        x_end = x + offset + 1
        for y in range(offset, n_y + offset):
            y_start = y - offset
            y_end = y + offset + 1
            for z in range(offset, n_z + offset):
                z_start = z - offset
                z_end = z + offset + 1
                new_value = 0
                weight_sum = 0
                # Coordinates of test pixel and patch bounds
                for i in range(max(- d, offset - x),
                              min(d + 1, n_x - x - 1)):
                    x_start_i = x_start + i
                    x_end_i = x_end + i
                    for j in range(max(- d, offset - y),
                                   min(d + 1, n_y - y - 1)):
                        y_start_j = y_start + j
                        y_end_j = y_end + j
                        for k in range(max(- d, offset - z),
                                   min(d + 1, n_z - z - 1)):
                            z_start_k = z_start + k
                            z_end_k = z_end + k
                            weight = patch_distance_3d(
                                    padded[x_start: x_end,
                                           y_start: y_end,
                                           z_start: z_end],
                                    padded[x_start_i: x_end_i,
                                           y_start_j: y_end_j,
                                           z_start_k: z_end_k],
                                    w, s)
                            weight_sum += weight
                            new_value += weight * padded[x + i, y + j, z + k]
                result[x, y, z] = new_value / weight_sum
    return result[offset:-offset, offset:-offset, offset:-offset]


@cython.cdivision(True)
@cython.boundscheck(False)
def _fast_nl_means_denoising_2d(image, int s=7, int d=13, float h=0.1):
    """
    Perform fast non-local means denoising on 2-D array, with the outer
    loop on patch shifts in order to reduce the number of operations.

    Parameters
    ----------
    image: ndarray
        2-D input data to be denoised

    s: int, optional
        size of patches used for denoising

    d: int, optional
        maximal distance in pixels where to search patches used for denoising

    h: float, optional
        cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.
    """
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef int offset = s / 2
    # Image padding: we need to account for patch size, possible shift,
    # + 1 for the boundary effects in finite differences
    cdef int pad_size = offset + d + 1
    cdef DTYPE_t [:, ::1] padded = np.ascontiguousarray(util.pad(image,
                                pad_size, mode='reflect').astype(np.float32))
    cdef DTYPE_t [:, ::1] result = np.zeros_like(padded)
    cdef DTYPE_t [:, ::1] weights = np.zeros_like(padded)
    cdef DTYPE_t [:, ::1] integral = np.zeros_like(padded)
    cdef int n_x, n_y, t1, t2, x, y
    cdef float weight, distance
    cdef float alpha
    cdef float h2 = h ** 2.
    cdef float s2 = s ** 2.
    n_x, n_y = image.shape
    n_x += 2 * pad_size
    n_y += 2 * pad_size
    # Outer loops on patch shifts
    # With t2 >= 0, reference patch is always on the left of test patch
    for t1 in range(-d, d + 1):
        for t2 in range(0, d + 1):
            # alpha is to account for patches on the same column
            # distance is computed twice in this case
            if t2 == 0 and t1 is not 0:
                alpha = 0.5
            else:
                alpha = 1.
            integral = np.zeros_like(padded)
            for x in range(max(1, - t1), min(n_x, n_x - t1)):
                for y in range(max(1, - t2), min(n_y, n_y - t2)):
                    integral[x, y] = (padded[x, y] -
                                      padded[x + t1, y + t2])**2 + \
                                      integral[x - 1, y] + integral[x, y - 1] \
                                      - integral[x - 1, y - 1]
            for x in range(max(offset, offset - t1),
                           min(n_x - offset, n_x - offset - t1)):
                for y in range(max(offset, offset - t2),
                               min(n_y - offset, n_y - offset - t2)):
                    distance = integral[x + offset, y + offset] + \
                               integral[x - offset, y - offset] - \
                               integral[x - offset, y + offset] - \
                               integral[x + offset, y - offset]
                    distance /= (s2 * h2)
                    if distance > 4:
                        continue
                    weight = alpha * exp(- distance)
                    weights[x, y] += weight
                    weights[x + t1, y + t2] += weight
                    result[x, y] += weight * padded[x + t1, y + t2]
                    result[x + t1, y + t2] += weight * padded[x, y]
    for x in range(offset, n_x - offset):
        for y in range(offset, n_y - offset):
            # I think there is no risk of division by zero
            # except in padded zone
            result[x, y] /= weights[x, y]
    return result[pad_size: - pad_size, pad_size: - pad_size]
