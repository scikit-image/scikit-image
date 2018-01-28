#cython: initializedcheck=False
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t IMGDTYPE

cdef double DISTANCE_CUTOFF = 5.0

cdef extern from "fast_exp.h":
    inline double fast_exp(double y) nogil


cdef inline double patch_distance_2d(IMGDTYPE [:, :] p1,
                                     IMGDTYPE [:, :] p2,
                                     IMGDTYPE [:, ::] w, int s, double var):
    """
    Compute a Gaussian distance between two image patches.

    Parameters
    ----------
    p1 : 2-D array_like
        First patch.
    p2 : 2-D array_like
        Second patch.
    w : 2-D array_like
        Array of weights for the different pixels of the patches.
    s : int
        Linear size of the patches.
    var : double
        Expected noise variance.

    Returns
    -------
    distance : double
        Gaussian distance between the two patches

    Notes
    -----
    The returned distance is given by

    .. math::  \exp( -w ((p1 - p2)^2 - 2*var))
    """
    cdef int i, j
    cdef int center = s / 2
    # Check if central pixel is too different in the 2 patches
    cdef double tmp_diff = p1[center, center] - p2[center, center]
    cdef double init = w[center, center] * tmp_diff * tmp_diff
    if init > 1:
        return 0.
    cdef double distance = 0
    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > DISTANCE_CUTOFF:
            return 0.
        for j in range(s):
            tmp_diff = p1[i, j] - p2[i, j]
            distance += w[i, j] * (tmp_diff * tmp_diff - 2 * var)
    distance = max(distance, 0)
    distance = fast_exp(-distance)
    return distance


cdef inline double patch_distance_2dmultichannel(IMGDTYPE [:, :, :] p1,
                                                 IMGDTYPE [:, :, :] p2,
                                                 IMGDTYPE [:, ::] w,
                                                 int s, double var,
                                                 int n_channels):
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
    var : double
        Expected noise variance.
    n_channels : int
        The number of channels.

    Returns
    -------
    distance : double
        Gaussian distance between the two patches

    Notes
    -----
    The returned distance is given by

    .. math::  \exp( -w ((p1 - p2)^2 - 2*var))
    """
    cdef int i, j, channel
    cdef double tmp_diff = 0
    cdef double distance = 0
    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > DISTANCE_CUTOFF:
            return 0.
        for j in range(s):
            for channel in range(n_channels):
                tmp_diff = p1[i, j, channel] - p2[i, j, channel]
                distance += w[i, j] * (tmp_diff * tmp_diff - 2 * var)
    distance = max(distance, 0)
    distance = fast_exp(-distance)
    return distance


cdef inline double patch_distance_3d(IMGDTYPE [:, :, :] p1,
                                     IMGDTYPE [:, :, :] p2,
                                     IMGDTYPE [:, :, ::] w, int s, double var):
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
    var : double
        Expected noise variance.

    Returns
    -------
    distance : double
        Gaussian distance between the two patches

    Notes
    -----
    The returned distance is given by

    .. math::  \exp( -w ((p1 - p2)^2 - 2*var))
    """
    cdef int i, j, k
    cdef double distance = 0
    cdef double tmp_diff
    for i in range(s):
        # exp of large negative numbers will be 0, so we'd better stop
        if distance > DISTANCE_CUTOFF:
            return 0.
        for j in range(s):
            for k in range(s):
                tmp_diff = p1[i, j, k] - p2[i, j, k]
                distance += w[i, j, k] * (tmp_diff * tmp_diff - 2 * var)
    distance = max(distance, 0.0)
    distance = fast_exp(-distance)
    return distance


def _nl_means_denoising_2d(image, int s=7, int d=13, double h=0.1,
                           double var=0.):
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
    h : double, optional
        Cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.
    var : double
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
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef int n_row, n_col, n_channels
    n_row, n_col, n_channels = image.shape
    cdef int offset = s / 2
    cdef int row, col, i, j, channel
    cdef int row_start, row_end, col_start, col_end
    cdef int row_start_i, row_end_i, col_start_j, col_end_j
    cdef IMGDTYPE [::1] new_values = np.zeros(n_channels).astype(np.float64)
    cdef IMGDTYPE [:, :, ::1] padded = np.ascontiguousarray(np.pad(image,
                       ((offset, offset), (offset, offset), (0, 0)),
                        mode='reflect').astype(np.float64))
    cdef IMGDTYPE [:, :, ::1] result = padded.copy()
    cdef double A = ((s - 1.) / 4.)
    cdef double new_value
    cdef double weight_sum, weight
    xg_row, xg_col = np.mgrid[-offset:offset + 1, -offset:offset + 1]
    cdef IMGDTYPE [:, ::1] w = np.ascontiguousarray(np.exp(
                             -(xg_row ** 2 + xg_col ** 2) / (2 * A ** 2)).
                             astype(np.float64))
    cdef double distance
    w = 1. / (n_channels * np.sum(w) * h ** 2) * w

    # Coordinates of central pixel
    # Iterate over rows, taking padding into account
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
                    # Shortcut for grayscale, else assume RGB
                    if n_channels == 1:
                        weight = patch_distance_2d(
                                 padded[row_start:row_end,
                                        col_start:col_end, 0],
                                 padded[row_start_i:row_end_i,
                                        col_start_j:col_end_j, 0],
                                 w, s, var)
                    else:
                        weight = patch_distance_2dmultichannel(
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


def _nl_means_denoising_3d(image, int s=7, int d=13, double h=0.1,
                           double var=0.0):
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
    h : double, optional
        Cut-off distance (in gray levels).
    var : double
        Expected noise variance.  If non-zero, this is used to reduce the
        apparent patch distances by the expected distance due to the noise.

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
    cdef IMGDTYPE [:, :, ::1] padded = np.ascontiguousarray(np.pad(
                                        image.astype(np.float64),
                                        offset, mode='reflect'))
    cdef IMGDTYPE [:, :, ::1] result = padded.copy()
    cdef double A = ((s - 1.) / 4.)
    cdef double new_value
    cdef double weight_sum, weight
    xg_pln, xg_row, xg_col = np.mgrid[-offset: offset + 1,
                                      -offset: offset + 1,
                                      -offset: offset + 1]
    cdef IMGDTYPE [:, :, ::1] w = np.ascontiguousarray(np.exp(
                            -(xg_pln ** 2 + xg_row ** 2 + xg_col ** 2) /
                             (2 * A ** 2)).astype(np.float64))
    cdef double distance
    cdef int pln, row, col, i, j, k
    cdef int pln_start, pln_end, row_start, row_end, col_start, col_end
    cdef int pln_start_i, pln_end_i, row_start_j, row_end_j, \
             col_start_k, col_end_k
    w = 1. / (np.sum(w) * h ** 2) * w

    # Coordinates of central pixel
    # Iterate over planes, taking padding into account
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
                            weight = patch_distance_3d(
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


cdef inline double _integral_to_distance_2d(IMGDTYPE [:, ::] integral, int row,
                                            int col, int offset, double h2s2):
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
    cdef double distance
    distance =  integral[row + offset, col + offset] + \
                integral[row - offset, col - offset] - \
                integral[row - offset, col + offset] - \
                integral[row + offset, col - offset]
    distance = max(distance, 0.0) / h2s2
    return distance


cdef inline double _integral_to_distance_3d(IMGDTYPE [:, :, ::] integral,
                                            int pln, int row, int col,
                                            int offset,
                                            double s_cube_h_square):
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
    cdef double distance
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


cdef inline _integral_image_2d(IMGDTYPE [:, :, ::] padded,
                               IMGDTYPE [:, ::] integral, int t_row,
                               int t_col, int n_row, int n_col, int n_channels,
                               double var):
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
    t_row : int
        Shift along the row axis.
    t_col : int
        Shift along the column axis.
    n_row : int
    n_col : int
    n_channels : int
    var : double
        Expected noise variance.  If non-zero, this is used to reduce the
        apparent patch distances by the expected distance due to the noise.

    Notes
    -----

    The integral computation could be performed using
    ``transform.integral_image``, but this helper function saves memory
    by avoiding copies of ``padded``.
    """
    cdef int row, col, channel
    cdef double distance
    for row in range(max(1, -t_row), min(n_row, n_row - t_row)):
        for col in range(max(1, -t_col), min(n_col, n_col - t_col)):
            if n_channels == 1:
                distance = (padded[row, col, 0] -
                            padded[row + t_row, col + t_col, 0])**2
                distance -= 2 * var
            else:
                distance = 0
                for channel in range(n_channels):
                    distance += (padded[row, col, channel] -
                                 padded[row + t_row, col + t_col, channel])**2
                distance -= 2 * n_channels * var
            integral[row, col] = distance + \
                                 integral[row - 1, col] + \
                                 integral[row, col - 1] - \
                                 integral[row - 1, col - 1]


cdef inline _integral_image_3d(IMGDTYPE [:, :, ::] padded,
                               IMGDTYPE [:, :, ::] integral, int t_pln,
                               int t_row, int t_col, int n_pln, int n_row,
                               int n_col,
                               double var):
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
    t_pln : int
        Shift along the plane axis.
    t_row : int
        Shift along the row axis.
    t_col : int
        Shift along the column axis.
    n_pln : int
    n_row : int
    n_col : int
    var : double
        Expected noise variance.  If non-zero, this is used to reduce the
        apparent patch distances by the expected distance due to the noise.

    Notes
    -----

    The integral computation could be performed using
    ``transform.integral_image``, but this helper function saves memory
    by avoiding copies of ``padded``.
    """
    cdef int pln, row, col
    cdef double distance
    for pln in range(max(1, -t_pln), min(n_pln, n_pln - t_pln)):
        for row in range(max(1, -t_row), min(n_row, n_row - t_row)):
            for col in range(max(1, -t_col), min(n_col, n_col - t_col)):
                distance = (padded[pln, row, col] -
                            padded[pln + t_pln, row + t_row, col + t_col])**2
                distance -= 2 * var
                integral[pln, row, col] = \
                    (distance +
                     integral[pln - 1, row, col] +
                     integral[pln, row - 1, col] +
                     integral[pln, row, col - 1] +
                     integral[pln - 1, row - 1, col - 1] -
                     integral[pln - 1, row - 1, col] -
                     integral[pln, row - 1, col - 1] -
                     integral[pln - 1, row, col - 1])


def _fast_nl_means_denoising_2d(image, int s=7, int d=13, double h=0.1,
                                double var=0.):
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
    h : double, optional
        Cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.
    var : double
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
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef int offset = s / 2
    # Image padding: we need to account for patch size, possible shift,
    # + 1 for the boundary effects in finite differences
    cdef int pad_size = offset + d + 1
    cdef IMGDTYPE [:, :, ::1] padded = np.ascontiguousarray(np.pad(image,
                          ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                          mode='reflect').astype(np.float64))
    cdef IMGDTYPE [:, :, ::1] result = np.zeros_like(padded)
    cdef IMGDTYPE [:, ::1] weights = np.zeros_like(padded[..., 0], order='C')
    cdef IMGDTYPE [:, ::1] integral = np.zeros_like(padded[..., 0], order='C')
    cdef int n_row, n_col, t_row, t_col, row, col, n_channels, channel
    cdef double weight, distance
    cdef double alpha
    cdef double h2 = h ** 2.
    cdef double s2 = s ** 2.
    n_row, n_col, n_channels = image.shape
    cdef double h2s2 = n_channels * h2 * s2
    n_row += 2 * pad_size
    n_col += 2 * pad_size

    # Outer loops on patch shifts
    # With t2 >= 0, reference patch is always on the left of test patch
    # Iterate over shifts along the row axis
    for t_row in range(-d, d + 1):
        # Iterate over shifts along the column axis
        for t_col in range(0, d + 1):
            # alpha is to account for patches on the same column
            # distance is computed twice in this case
            if t_col == 0 and t_row is not 0:
                alpha = 0.5
            else:
                alpha = 1.
            # Compute integral image of the squared difference between
            # padded and the same image shifted by (t_row, t_col)
            integral = np.zeros_like(padded[..., 0], order='C')
            _integral_image_2d(padded, integral, t_row, t_col,
                               n_row, n_col, n_channels, var)

            # Inner loops on pixel coordinates
            # Iterate over rows, taking offset and shift into account
            for row in range(max(offset, offset - t_row),
                             min(n_row - offset, n_row - offset - t_row)):
                # Iterate over columns, taking offset and shift into account
                for col in range(max(offset, offset - t_col),
                                 min(n_col - offset, n_col - offset - t_col)):
                    # Compute squared distance between shifted patches
                    distance = _integral_to_distance_2d(integral, row, col,
                                                     offset, h2s2)
                    # exp of large negative numbers is close to zero
                    if distance > DISTANCE_CUTOFF:
                        continue
                    weight = alpha * fast_exp(-distance)
                    # Accumulate weights corresponding to different shifts
                    weights[row, col] += weight
                    weights[row + t_row, col + t_col] += weight
                    # Iterate over channels
                    for channel in range(n_channels):
                        result[row, col, channel] += weight * \
                                    padded[row + t_row, col + t_col, channel]
                        result[row + t_row, col + t_col, channel] += \
                                        weight * padded[row, col, channel]

    # Normalize pixel values using sum of weights of contributing patches
    for row in range(offset, n_row - offset):
        for col in range(offset, n_col - offset):
            for channel in range(n_channels):
                # No risk of division by zero, since the contribution
                # of a null shift is strictly positive
                result[row, col, channel] /= weights[row, col]

    # Return cropped result, undoing padding
    return result[pad_size:-pad_size, pad_size:-pad_size]


def _fast_nl_means_denoising_3d(image, int s=5, int d=7, double h=0.1,
                                double var=0.):
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
    h : double, optional
        cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.
    var : double
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
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef int offset = s / 2
    # Image padding: we need to account for patch size, possible shift,
    # + 1 for the boundary effects in finite differences
    cdef int pad_size = offset + d + 1
    cdef IMGDTYPE [:, :, ::1] padded = np.ascontiguousarray(np.pad(image,
                                pad_size, mode='reflect').astype(np.float64))
    cdef IMGDTYPE [:, :, ::1] result = np.zeros_like(padded)
    cdef IMGDTYPE [:, :, ::1] weights = np.zeros_like(padded)
    cdef IMGDTYPE [:, :, ::1] integral = np.zeros_like(padded)
    cdef int n_pln, n_row, n_col, t_pln, t_row, t_col, \
             pln, row, col
    cdef int pln_dist_min, pln_dist_max, row_dist_min, row_dist_max, \
             col_dist_min, col_dist_max
    cdef double weight, distance
    cdef double alpha
    cdef double h_square = h ** 2.0
    cdef double s_cube = s ** 3.0
    cdef double s_cube_h_square = h_square * s_cube
    n_pln, n_row, n_col = image.shape
    n_pln += 2 * pad_size
    n_row += 2 * pad_size
    n_col += 2 * pad_size

    # Outer loops on patch shifts
    # With t2 >= 0, reference patch is always on the left of test patch
    # Iterate over shifts along the plane axis
    for t_pln in range(-d, d + 1):
        pln_dist_min = max(offset, offset - t_pln)
        pln_dist_max = min(n_pln - offset, n_pln - offset - t_pln)
        # Iterate over shifts along the row axis
        for t_row in range(-d, d + 1):
            row_dist_min = max(offset, offset - t_row)
            row_dist_max = min(n_row - offset, n_row - offset - t_row)
            # Iterate over shifts along the column axis
            for t_col in range(0, d + 1):
                col_dist_min = max(offset, offset - t_col)
                col_dist_max = min(n_col - offset, n_col - offset - t_col)
                # alpha is to account for patches on the same column
                # distance is computed twice in this case
                if t_col == 0 and (t_pln is not 0 or t_row is not 0):
                    alpha = 0.5
                else:
                    alpha = 1.0

                # Compute integral image of the squared difference between
                # padded and the same image shifted by (t_pln, t_row, t_col)
                integral = np.zeros_like(padded)
                _integral_image_3d(padded, integral, t_pln, t_row, t_col,
                                   n_pln, n_row, n_col, var)

                # Inner loops on pixel coordinates
                # Iterate over planes, taking offset and shift into account
                for pln in range(pln_dist_min, pln_dist_max):
                    # Iterate over rows, taking offset and shift into account
                    for row in range(row_dist_min, row_dist_max):
                        # Iterate over columns
                        for col in range(col_dist_min, col_dist_max):
                            # Compute squared distance between shifted patches
                            distance = _integral_to_distance_3d(integral,
                                        pln, row, col, offset, s_cube_h_square)
                            # exp of large negative numbers is close to zero
                            if distance > DISTANCE_CUTOFF:
                                continue

                            weight = alpha * fast_exp(-distance)
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

    # Normalize pixel values using sum of weights of contributing patches
    for pln in range(offset, n_pln - offset):
        for row in range(offset, n_row - offset):
            for col in range(offset, n_col - offset):
                # No risk of division by zero, since the contribution
                # of a null shift is strictly positive
                result[pln, row, col] /= weights[pln, row, col]

    # Return cropped result, undoing padding
    return result[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]
