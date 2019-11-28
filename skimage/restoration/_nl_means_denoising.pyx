#cython: initializedcheck=False
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp

from .._shared.fused_numerics cimport np_floats

cdef extern from "fast_exp.h":
    double fast_exp(double y) nogil
    float fast_expf(float y) nogil


cdef inline np_floats exp_func(np_floats x) nogil:
    if np_floats is cnp.float32_t:
        return fast_expf(x)
    else:
        return fast_exp(x)


cdef inline np_floats patch_distance_2d(np_floats [:, :, :] p1,
                                        np_floats [:, :, :] p2,
                                        np_floats [:, ::] w,
                                        Py_ssize_t s, np_floats var,
                                        Py_ssize_t n_channels) nogil:
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
    s : Py_ssize_t
        Linear size of the patches.
    var : np_floats
        Expected noise variance.
    n_channels : Py_ssize_t
        The number of channels.

    Returns
    -------
    distance : np_floats
        Gaussian distance between the two patches

    Notes
    -----
    The returned distance is given by

    .. math::  \exp( -w ((p1 - p2)^2 - 2*var))
    """

    cdef Py_ssize_t i, j, channel
    cdef np_floats DISTANCE_CUTOFF = 5.0
    cdef np_floats tmp_diff = 0
    cdef np_floats distance = 0
    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > DISTANCE_CUTOFF:
            return 0.
        for j in range(s):
            for channel in range(n_channels):
                tmp_diff = p1[i, j, channel] - p2[i, j, channel]
                distance += w[i, j] * (tmp_diff * tmp_diff - 2 * var)
    return exp_func(-max(0.0, distance))


cdef inline np_floats patch_distance_3d(np_floats [:, :, :] p1,
                                        np_floats [:, :, :] p2,
                                        np_floats [:, :, ::] w,
                                        Py_ssize_t s, np_floats var) nogil:
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
    s : Py_ssize_t
        Linear size of the patches.
    var : np_floats
        Expected noise variance.

    Returns
    -------
    distance : np_floats
        Gaussian distance between the two patches

    Notes
    -----
    The returned distance is given by

    .. math::  \exp( -w ((p1 - p2)^2 - 2*var))
    """

    cdef Py_ssize_t i, j, k
    cdef np_floats DISTANCE_CUTOFF = 5.0
    cdef np_floats distance = 0
    cdef np_floats tmp_diff
    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > DISTANCE_CUTOFF:
            return 0.
        for j in range(s):
            for k in range(s):
                tmp_diff = p1[i, j, k] - p2[i, j, k]
                distance += w[i, j, k] * (tmp_diff * tmp_diff - 2 * var)
    return exp_func(-max(0.0, distance))


def _nl_means_denoising_2d(cnp.ndarray[np_floats, ndim=3] image, Py_ssize_t s=7,
                           Py_ssize_t d=13, np_floats h=0.1, np_floats var=0.):
    """
    Perform non-local means denoising on 2-D RGB image

    Parameters
    ----------
    image : ndarray
        Input RGB image to be denoised
    s : Py_ssize_t, optional
        Size of patches used for denoising
    d : Py_ssize_t, optional
        Maximal distance in pixels where to search patches used for denoising
    h : np_floats, optional
        Cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.
    var : np_floats
        Expected noise variance.  If non-zero, this is used to reduce the
        apparent patch distances by the expected distance due to the noise.

    Notes
    -----
    This function operates on 2D grayscale and multichannel images.  For
    2D grayscale images, the input should be 3D with size 1 along the last
    axis.  The code is compatible with an arbitrary number of channels.

    Returns
    -------
    result : ndarray
        Denoised image, of same shape as input image.
    """

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef Py_ssize_t n_row, n_col, n_channels
    n_row, n_col, n_channels = image.shape[0], image.shape[1], image.shape[2]
    cdef Py_ssize_t offset = s / 2
    cdef Py_ssize_t row, col, i, j, channel
    cdef Py_ssize_t row_start, row_end, col_start, col_end
    cdef Py_ssize_t row_start_i, row_end_i, col_start_j, col_end_j
    cdef np_floats[::1] new_values = np.zeros(n_channels, dtype=dtype)
    cdef np_floats[:, :, ::1] padded = np.ascontiguousarray(
        np.pad(image, ((offset, offset), (offset, offset), (0, 0)),
               mode='reflect'))
    cdef np_floats [:, :, ::1] result = padded.copy()
    cdef np_floats A = ((s - 1.) / 4.)
    cdef np_floats new_value
    cdef np_floats weight_sum, weight
    cdef np_floats [::] range_vals = np.arange(-offset, offset + 1,
                                               dtype=dtype)
    xg_row, xg_col = np.meshgrid(range_vals, range_vals, indexing='ij')
    cdef np_floats [:, ::1] w = np.ascontiguousarray(
        np.exp(-(xg_row * xg_row + xg_col * xg_col) / (2 * A * A)))
    w *= 1. / (n_channels * np.sum(w) * h * h)

    # Coordinates of central pixel
    # Iterate over rows, taking padding into account
    with nogil:
        for row in range(offset, n_row + offset):
            row_start = row - offset
            row_end = row + offset + 1
            # Iterate over columns, taking padding into account
            for col in range(offset, n_col + offset):
                # Initialize per-channel bins
                for channel in range(n_channels):
                    new_values[channel] = 0
                # Reset weights for each local region
                weight_sum = 0
                col_start = col - offset
                col_end = col + offset + 1

                # Iterate over local 2d patch for each pixel
                # First rows
                for i in range(max(-d, offset - row),
                               min(d + 1, n_row + offset - row)):
                    row_start_i = row_start + i
                    row_end_i = row_end + i
                    # Local patch columns
                    for j in range(max(-d, offset - col),
                                   min(d + 1, n_col + offset - col)):
                        col_start_j = col_start + j
                        col_end_j = col_end + j
                        weight = patch_distance_2d[np_floats](
                            padded[row_start:row_end,
                                   col_start:col_end, :],
                            padded[row_start_i:row_end_i,
                                   col_start_j:col_end_j, :],
                            w, s, var, n_channels)

                        # Collect results in weight sum
                        weight_sum += weight
                        # Apply to each channel multiplicatively
                        for channel in range(n_channels):
                            new_values[channel] += weight * padded[row + i,
                                                                   col + j,
                                                                   channel]

                # Normalize the result
                for channel in range(n_channels):
                    result[row, col, channel] = new_values[channel] / weight_sum

    # Return cropped result, undoing padding
    return result[offset:-offset, offset:-offset]


def _nl_means_denoising_3d(cnp.ndarray[np_floats, ndim=3] image,
                           Py_ssize_t s=7, Py_ssize_t d=13,
                           np_floats h=0.1, np_floats var=0.0):
    """
    Perform non-local means denoising on 3-D array

    Parameters
    ----------
    image : ndarray
        Input data to be denoised.
    s : int, optional
        Size of patches used for denoising.
    d : Py_ssize_t, optional
        Maximal distance in pixels where to search patches used for denoising.
    h : np_floats, optional
        Cut-off distance (in gray levels).
    var : np_floats
        Expected noise variance.  If non-zero, this is used to reduce the
        apparent patch distances by the expected distance due to the noise.

    Returns
    -------
    result : ndarray
        Denoised image, of same shape as input image.
    """

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef Py_ssize_t n_pln, n_row, n_col
    n_pln, n_row, n_col = image.shape[0], image.shape[1], image.shape[2]
    cdef Py_ssize_t offset = s / 2
    # padd the image so that boundaries are denoised as well
    cdef np_floats [:, :, ::1] padded = np.ascontiguousarray(
        np.pad(image, offset, mode='reflect'))
    cdef np_floats [:, :, ::1] result = padded.copy()
    cdef np_floats A = ((s - 1.) / 4.)
    cdef np_floats new_value
    cdef np_floats weight_sum, weight
    cdef np_floats [::] range_vals = np.arange(-offset, offset + 1,
                                               dtype=dtype)
    xg_pln, xg_row, xg_col = np.meshgrid(range_vals, range_vals, range_vals,
                                         indexing='ij')
    cdef np_floats [:, :, ::1] w = np.ascontiguousarray(
        np.exp(-(xg_pln * xg_pln + xg_row * xg_row + xg_col * xg_col) /
               (2 * A * A)))
    cdef Py_ssize_t pln, row, col, i, j, k
    cdef Py_ssize_t pln_start, pln_end, row_start, row_end, col_start, col_end
    cdef Py_ssize_t pln_start_i, pln_end_i, row_start_j, row_end_j, \
             col_start_k, col_end_k
    w *= 1. / (np.sum(w) * h * h)

    # Coordinates of central pixel
    # Iterate over planes, taking padding into account
    with nogil:
        for pln in range(offset, n_pln + offset):
            pln_start = pln - offset
            pln_end = pln + offset + 1
            # Iterate over rows, taking padding into account
            for row in range(offset, n_row + offset):
                row_start = row - offset
                row_end = row + offset + 1
                # Iterate over columns, taking padding into account
                for col in range(offset, n_col + offset):
                    col_start = col - offset
                    col_end = col + offset + 1
                    new_value = 0
                    weight_sum = 0

                    # Iterate over local 3d patch for each pixel
                    # First planes
                    for i in range(max(-d, offset - pln),
                                   min(d + 1, n_pln + offset - pln)):
                        pln_start_i = pln_start + i
                        pln_end_i = pln_end + i
                        # Rows
                        for j in range(max(-d, offset - row),
                                       min(d + 1, n_row + offset - row)):
                            row_start_j = row_start + j
                            row_end_j = row_end + j
                            # Columns
                            for k in range(max(-d, offset - col),
                                           min(d + 1, n_col + offset - col)):
                                col_start_k = col_start + k
                                col_end_k = col_end + k
                                weight = patch_distance_3d[np_floats](
                                        padded[pln_start:pln_end,
                                               row_start:row_end,
                                               col_start:col_end],
                                        padded[pln_start_i:pln_end_i,
                                               row_start_j:row_end_j,
                                               col_start_k:col_end_k],
                                        w, s, var)
                                # Collect results in weight sum
                                weight_sum += weight
                                new_value += weight * padded[pln + i,
                                                             row + j, col + k]

                    # Normalize the result
                    result[pln, row, col] = new_value / weight_sum

    # Return cropped result, undoing padding
    return result[offset:-offset, offset:-offset, offset:-offset]

#-------------- Accelerated algorithm of Froment 2015 ------------------


cdef inline np_floats _integral_to_distance_2d(np_floats [:, ::] integral,
                                               Py_ssize_t row, Py_ssize_t col,
                                               Py_ssize_t offset,
                                               np_floats h2s2) nogil:
    """
    References
    ----------
    J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen, Fast
    nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
    International Symposium on Biomedical Imaging: From Nano to Macro,
    2008, pp. 1331-1334.

    Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
    Denoising. Image Processing On Line, 2014, vol. 4, p. 300-326.

    Used in _fast_nl_means_denoising_2d
    """
    cdef np_floats distance
    distance =  integral[row + offset, col + offset] + \
                integral[row - offset, col - offset] - \
                integral[row - offset, col + offset] - \
                integral[row + offset, col - offset]
    distance = max(distance, 0.0) / h2s2
    return distance


cdef inline np_floats _integral_to_distance_3d(np_floats [:, :, ::]
                                               integral, Py_ssize_t pln,
                                               Py_ssize_t row, Py_ssize_t col,
                                               Py_ssize_t offset,
                                               np_floats s_cube_h_square) nogil:
    """
    References
    ----------
    J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen, Fast
    nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
    International Symposium on Biomedical Imaging: From Nano to Macro,
    2008, pp. 1331-1334.

    Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
    Denoising. Image Processing On Line, 2014, vol. 4, p. 300-326.

    Used in _fast_nl_means_denoising_3d
    """
    cdef np_floats distance
    distance = (integral[pln + offset, row + offset, col + offset] -
                integral[pln - offset, row - offset, col - offset] +
                integral[pln - offset, row - offset, col + offset] +
                integral[pln - offset, row + offset, col - offset] +
                integral[pln + offset, row - offset, col - offset] -
                integral[pln - offset, row + offset, col + offset] -
                integral[pln + offset, row - offset, col + offset] -
                integral[pln + offset, row + offset, col - offset])
    distance = max(distance, 0.0) / (s_cube_h_square)
    return distance


cdef inline void _integral_image_2d(np_floats [:, :, ::] padded,
                                    np_floats [:, ::] integral,
                                    Py_ssize_t t_row, Py_ssize_t
                                    t_col, Py_ssize_t n_row,
                                    Py_ssize_t n_col, Py_ssize_t n_channels,
                                    np_floats var) nogil:
    """
    Computes the integral of the squared difference between an image ``padded``
    and the same image shifted by ``(t_row, t_col)``.

    Parameters
    ----------
    padded : ndarray of shape (n_row, n_col, n_channels)
        Image of interest.
    integral : ndarray
        Output of the function. The array is filled with integral values.
        ``integral`` should have the same shape as ``padded``.
    t_row : Py_ssize_t
        Shift along the row axis.
    t_col : Py_ssize_t
        Shift along the column axis.
    n_row : Py_ssize_t
    n_col : Py_ssize_t
    n_channels : Py_ssize_t
    var : np_floats
        Expected noise variance.  If non-zero, this is used to reduce the
        apparent patch distances by the expected distance due to the noise.

    Notes
    -----

    The integral computation could be performed using
    ``transform.integral_image``, but this helper function saves memory
    by avoiding copies of ``padded``.
    """
    cdef Py_ssize_t row, col, channel
    cdef np_floats distance, t
    var *= 2.0

    for row in range(max(1, -t_row), min(n_row, n_row - t_row)):
        for col in range(max(1, -t_col), min(n_col, n_col - t_col)):
            distance = 0
            for channel in range(n_channels):
                t = (padded[row, col, channel] -
                     padded[row + t_row, col + t_col, channel])
                distance += t * t
            distance -= n_channels * var
            integral[row, col] = (distance +
                                  integral[row - 1, col] +
                                  integral[row, col - 1] -
                                  integral[row - 1, col - 1])


cdef inline void _integral_image_3d(np_floats [:, :, ::] padded,
                                    np_floats [:, :, ::] integral,
                                    Py_ssize_t t_pln, Py_ssize_t
                                    t_row, Py_ssize_t t_col,
                                    Py_ssize_t n_pln, Py_ssize_t n_row,
                                    Py_ssize_t n_col, np_floats
                                    var) nogil:
    """
    Computes the integral of the squared difference between an image ``padded``
    and the same image shifted by ``(t_pln, t_row, t_col)``.

    Parameters
    ----------
    padded : ndarray of shape (n_pln, n_row, n_col)
        Image of interest.
    integral : ndarray
        Output of the function. The array is filled with integral values.
        ``integral`` should have the same shape as ``padded``.
    t_pln : Py_ssize_t
        Shift along the plane axis.
    t_row : Py_ssize_t
        Shift along the row axis.
    t_col : Py_ssize_t
        Shift along the column axis.
    n_pln : Py_ssize_t
    n_row : Py_ssize_t
    n_col : Py_ssize_t
    var : np_floats
        Expected noise variance.  If non-zero, this is used to reduce the
        apparent patch distances by the expected distance due to the noise.

    Notes
    -----

    The integral computation could be performed using
    ``transform.integral_image``, but this helper function saves memory
    by avoiding copies of ``padded``.
    """
    cdef Py_ssize_t pln, row, col
    cdef np_floats distance
    var *= 2.0
    for pln in range(max(1, -t_pln), min(n_pln, n_pln - t_pln)):
        for row in range(max(1, -t_row), min(n_row, n_row - t_row)):
            for col in range(max(1, -t_col), min(n_col, n_col - t_col)):
                distance = (padded[pln, row, col] -
                            padded[pln + t_pln, row + t_row, col + t_col])
                distance *= distance
                distance -= var
                integral[pln, row, col] = (
                     distance +
                     integral[pln - 1, row, col] +
                     integral[pln, row - 1, col] +
                     integral[pln, row, col - 1] +
                     integral[pln - 1, row - 1, col - 1] -
                     integral[pln - 1, row - 1, col] -
                     integral[pln, row - 1, col - 1] -
                     integral[pln - 1, row, col - 1])


def _fast_nl_means_denoising_2d(cnp.ndarray[np_floats, ndim=3] image,
                                Py_ssize_t s=7, Py_ssize_t d=13,
                                np_floats h=0.1, np_floats var=0.):
    """
    Perform fast non-local means denoising on 2-D array, with the outer
    loop on patch shifts in order to reduce the number of operations.

    Parameters
    ----------
    image : ndarray
        2-D input data to be denoised, grayscale or RGB.
    s : Py_ssize_t, optional
        Size of patches used for denoising.
    d : Py_ssize_t, optional
        Maximal distance in pixels where to search patches used for denoising.
    h : np_floats, optional
        Cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.
    var : np_floats
        Expected noise variance.  If non-zero, this is used to reduce the
        apparent patch distances by the expected distance due to the noise.

    Returns
    -------
    result : ndarray
        Denoised image, of same shape as input image.

    References
    ----------
    J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen, Fast
    nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
    International Symposium on Biomedical Imaging: From Nano to Macro,
    2008, pp. 1331-1334.

    Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
    Denoising. Image Processing On Line, 2014, vol. 4, p. 300-326.
    """

    cdef np_floats DISTANCE_CUTOFF = 5.0
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef Py_ssize_t offset = s / 2
    # Image padding: we need to account for patch size, possible shift,
    # + 1 for the boundary effects in finite differences
    cdef Py_ssize_t pad_size = offset + d + 1
    cdef np_floats [:, :, ::1] padded = np.ascontiguousarray(
        np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
               mode='reflect'))
    cdef np_floats [:, :, ::1] result = np.zeros_like(padded)
    cdef np_floats [:, ::1] weights = np.zeros_like(padded[..., 0], order='C')
    cdef np_floats [:, ::1] integral = np.zeros_like(padded[..., 0], order='C')
    cdef Py_ssize_t n_row, n_col, t_row, t_col, row, col, n_channels, channel
    cdef np_floats weight, distance
    cdef np_floats alpha
    n_row, n_col, n_channels = image.shape[0], image.shape[1], image.shape[2]
    cdef np_floats h2s2 = n_channels * h * h * s * s
    n_row += 2 * pad_size
    n_col += 2 * pad_size

    with nogil:
        # Outer loops on patch shifts
        # With t2 >= 0, reference patch is always on the left of test patch
        # Iterate over shifts along the row axis
        for t_row in range(-d, d + 1):
            # alpha is to account for patches on the same column
            # distance is computed twice in this case
            if t_row == 0:
                alpha = 1.0
            else:
                alpha = 0.5
            # Iterate over shifts along the column axis
            for t_col in range(0, d + 1):
                # Compute integral image of the squared difference between
                # padded and the same image shifted by (t_row, t_col)
                _integral_image_2d[np_floats](padded, integral, t_row, t_col,
                                              n_row, n_col, n_channels, var)

                # Inner loops on pixel coordinates
                # Iterate over rows, taking offset and shift into account
                for row in range(max(offset, offset - t_row),
                                 min(n_row - offset, n_row - offset - t_row)):
                    # Iterate over columns, taking offset and shift into account
                    for col in range(
                            max(offset, offset - t_col),
                            min(n_col - offset, n_col - offset - t_col)):
                        # Compute squared distance between shifted patches
                        distance = _integral_to_distance_2d[np_floats](
                            integral, row, col, offset, h2s2)
                        # exp of large negative numbers is close to zero
                        if distance > DISTANCE_CUTOFF:
                            continue
                        weight = alpha * exp_func(-distance)
                        # Accumulate weights corresponding to different shifts
                        weights[row, col] += weight
                        weights[row + t_row, col + t_col] += weight
                        # Iterate over channels
                        for channel in range(n_channels):
                            result[row, col, channel] += weight * \
                                padded[row + t_row, col + t_col, channel]
                            result[row + t_row, col + t_col, channel] += \
                                weight * padded[row, col, channel]
                alpha = 1

        # Normalize pixel values using sum of weights of contributing patches
        for row in range(offset, n_row - offset):
            for col in range(offset, n_col - offset):
                for channel in range(n_channels):
                    # No risk of division by zero, since the contribution
                    # of a null shift is strictly positive
                    result[row, col, channel] /= weights[row, col]

    # Return cropped result, undoing padding
    return result[pad_size:-pad_size, pad_size:-pad_size]


def _fast_nl_means_denoising_3d(cnp.ndarray[np_floats, ndim=3] image,
                                Py_ssize_t s=5, Py_ssize_t d=7, np_floats h=0.1,
                                np_floats var=0.):
    """
    Perform fast non-local means denoising on 3-D array, with the outer
    loop on patch shifts in order to reduce the number of operations.

    Parameters
    ----------
    image : ndarray
        3-D input data to be denoised.
    s : Py_ssize_t, optional
        Size of patches used for denoising.
    d : Py_ssize_t, optional
        Maximal distance in pixels where to search patches used for denoising.
    h : np_floats, optional
        cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.
    var : np_floats
        Expected noise variance.  If non-zero, this is used to reduce the
        apparent patch distances by the expected distance due to the noise.

    Returns
    -------
    result : ndarray
        Denoised image, of same shape as input image.

    References
    ----------
    J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen, Fast
    nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
    International Symposium on Biomedical Imaging: From Nano to Macro,
    2008, pp. 1331-1334.

    Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
    Denoising. Image Processing On Line, 2014, vol. 4, p. 300-326.
    """

    cdef np_floats DISTANCE_CUTOFF = 5.0
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef Py_ssize_t offset = s / 2
    # Image padding: we need to account for patch size, possible shift,
    # + 1 for the boundary effects in finite differences
    cdef Py_ssize_t pad_size = offset + d + 1
    cdef np_floats [:, :, ::1] padded = np.ascontiguousarray(
        np.pad(image, pad_size, mode='reflect'))
    cdef np_floats [:, :, ::1] result = np.zeros_like(padded)
    cdef np_floats [:, :, ::1] weights = np.zeros_like(padded)
    cdef np_floats [:, :, ::1] integral = np.zeros_like(padded)
    cdef Py_ssize_t n_pln, n_row, n_col, t_pln, t_row, t_col, \
             pln, row, col
    cdef Py_ssize_t pln_dist_min, pln_dist_max, row_dist_min, row_dist_max, \
             col_dist_min, col_dist_max
    cdef np_floats weight, distance
    cdef np_floats alpha
    cdef np_floats s_cube_h_square = h * h * s * s * s
    n_pln, n_row, n_col = image.shape[0], image.shape[1], image.shape[2]
    n_pln += 2 * pad_size
    n_row += 2 * pad_size
    n_col += 2 * pad_size

    with nogil:
        # Outer loops on patch shifts
        # With t2 >= 0, reference patch is always on the left of test patch
        # Iterate over shifts along the plane axis
        for t_pln in range(-d, d + 1):
            pln_dist_min = max(offset, offset - t_pln)
            pln_dist_max = min(n_pln - offset, n_pln - offset - t_pln)
                    # alpha is to account for patches on the same column
                    # distance is computed twice in this case
            if t_pln == 0:
                alpha = 1.0
            else:
                alpha = 0.5
            # Iterate over shifts along the row axis
            for t_row in range(-d, d + 1):
                row_dist_min = max(offset, offset - t_row)
                row_dist_max = min(n_row - offset, n_row - offset - t_row)
                if t_row == 0:
                    alpha = 1.0
                else:
                    alpha = 0.5
                # Iterate over shifts along the column axis
                for t_col in range(0, d + 1):
                    col_dist_min = max(offset, offset - t_col)
                    col_dist_max = min(n_col - offset, n_col - offset - t_col)

                    # Compute integral image of the squared difference between
                    # padded and the same image shifted by (t_pln, t_row, t_col)
                    _integral_image_3d[np_floats](
                        padded, integral, t_pln, t_row, t_col,
                        n_pln, n_row, n_col, var)

                    # Inner loops on pixel coordinates
                    # Iterate over planes, taking offset and shift into account
                    for pln in range(pln_dist_min, pln_dist_max):
                        # Iterate over rows, taking offset and shift
                        # into account
                        for row in range(row_dist_min, row_dist_max):
                            # Iterate over columns
                            for col in range(col_dist_min, col_dist_max):
                                # Compute squared distance between
                                # shifted patches
                                distance = _integral_to_distance_3d[np_floats](
                                    integral, pln, row, col, offset,
                                    s_cube_h_square)
                                # exp of large negative numbers is close to zero
                                if distance > DISTANCE_CUTOFF:
                                    continue

                                weight = alpha * exp_func(-distance)
                                # Accumulate weights for the different shifts
                                weights[pln, row, col] += weight
                                weights[pln + t_pln, row + t_row,
                                                     col + t_col] += weight
                                result[pln, row, col] += weight * \
                                        padded[pln + t_pln, row + t_row,
                                                            col + t_col]
                                result[pln + t_pln, row + t_row,
                                                    col + t_col] += weight * \
                                                      padded[pln, row, col]
                    alpha = 1.0

        # Normalize pixel values using sum of weights of contributing patches
        for pln in range(offset, n_pln - offset):
            for row in range(offset, n_row - offset):
                for col in range(offset, n_col - offset):
                    # No risk of division by zero, since the contribution
                    # of a null shift is strictly positive
                    result[pln, row, col] /= weights[pln, row, col]

    # Return cropped result, undoing padding
    return result[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]
