#cython: initializedcheck=False
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp

from .._shared.fast_exp cimport _fast_exp
from .._shared.fused_numerics cimport np_floats


cnp.import_array()

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
    var_diff : np_floats
        The double of the expected noise variance.
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
                distance += w[i, j] * (tmp_diff * tmp_diff - var)
    return _fast_exp(-max(0.0, distance))


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
    var_diff : np_floats
        The double of the expected noise variance.

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
                distance += w[i, j, k] * (tmp_diff * tmp_diff - var)
    return _fast_exp(-max(0.0, distance))


def _nl_means_denoising_2d(cnp.ndarray[np_floats, ndim=3] image, Py_ssize_t s,
                           Py_ssize_t d, double h, double var):
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

    if s % 2 == 0:
        s += 1  # odd value for symmetric patch

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef Py_ssize_t n_row, n_col, n_channels
    n_row, n_col, n_channels = image.shape[0], image.shape[1], image.shape[2]
    cdef Py_ssize_t offset = s / 2
    cdef Py_ssize_t row, col, i, j, channel, i_start, i_end, j_start, j_end
    cdef np_floats[::1] new_values = np.zeros(n_channels, dtype=dtype)
    cdef np_floats[:, :, ::1] padded = np.ascontiguousarray(
        np.pad(image, ((offset, offset), (offset, offset), (0, 0)),
               mode='reflect'))
    cdef np_floats [:, :, ::1] result = np.empty_like(image)
    cdef np_floats new_value
    cdef np_floats weight_sum, weight

    cdef np_floats A = ((s - 1.) / 4.)
    cdef np_floats [::1] range_vals = np.arange(-offset, offset + 1,
                                                dtype=dtype)
    xg_row, xg_col = np.meshgrid(range_vals, range_vals, indexing='ij')
    cdef np_floats [:, ::1] w = np.ascontiguousarray(
        np.exp(-(xg_row * xg_row + xg_col * xg_col) / (2 * A * A)))
    w *= 1. / (n_channels * np.sum(w) * h * h)

    cdef np_floats [:, :, :] central_patch
    var *= 2

    # Iterate over rows, taking padding into account
    with nogil:
        for row in range(n_row):
            # Iterate over columns, taking padding into account
            i_start = row - min(d, row)
            i_end = row + min(d + 1, n_row - row)

            for col in range(n_col):
                # Initialize per-channel bins
                new_values[:] = 0
                # Reset weights for each local region
                weight_sum = 0

                central_patch = padded[row:row+s, col:col+s, :]
                j_start = col - min(d, col)
                j_end = col + min(d + 1, n_col - col)

                # Iterate over local 2d patch for each pixel
                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        weight = patch_distance_2d[np_floats](
                            central_patch,
                            padded[i:i+s, j:j+s, :],
                            w, s, var, n_channels)

                        # Collect results in weight sum
                        weight_sum += weight
                        # Apply to each channel multiplicatively
                        for channel in range(n_channels):
                            new_values[channel] += weight * padded[i+offset,
                                                                   j+offset,
                                                                   channel]

                # Normalize the result
                for channel in range(n_channels):
                    result[row, col, channel] = new_values[channel] / weight_sum

    return np.squeeze(np.asarray(result))


def _nl_means_denoising_3d(cnp.ndarray[np_floats, ndim=3] image,
                           Py_ssize_t s, Py_ssize_t d,
                           double h, double var):
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

    if s % 2 == 0:
        s += 1  # odd value for symmetric patch

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef Py_ssize_t n_pln, n_row, n_col
    n_pln, n_row, n_col = image.shape[0], image.shape[1], image.shape[2]
    cdef Py_ssize_t i_start, i_end, j_start, j_end, k_start, k_end
    cdef Py_ssize_t pln, row, col, i, j, k
    cdef Py_ssize_t offset = s / 2
    # padd the image so that boundaries are denoised as well
    cdef np_floats [:, :, ::1] padded = np.ascontiguousarray(
        np.pad(image, offset, mode='reflect'))
    cdef np_floats [:, :, ::1] result = np.empty_like(image)
    cdef np_floats new_value
    cdef np_floats weight_sum, weight

    cdef np_floats A = ((s - 1.) / 4.)
    cdef np_floats [::] range_vals = np.arange(-offset, offset + 1,
                                               dtype=dtype)
    xg_pln, xg_row, xg_col = np.meshgrid(range_vals, range_vals, range_vals,
                                         indexing='ij')
    cdef np_floats [:, :, ::1] w = np.ascontiguousarray(
        np.exp(-(xg_pln * xg_pln + xg_row * xg_row + xg_col * xg_col) /
               (2 * A * A)))
    w *= 1. / (np.sum(w) * h * h)

    cdef np_floats [:, :, :] central_patch
    var *= 2

    # Iterate over planes, taking padding into account
    with nogil:
        for pln in range(n_pln):
            i_start = pln - min(d, pln)
            i_end = pln + min(d + 1, n_pln - pln)
            # Iterate over rows, taking padding into account
            for row in range(n_row):
                j_start = row - min(d, row)
                j_end = row + min(d + 1, n_row - row)
                # Iterate over columns, taking padding into account
                for col in range(n_col):
                    k_start = col - min(d, col)
                    k_end = col + min(d + 1, n_col - col)

                    central_patch = padded[pln:pln+s, row:row+s, col:col+s]

                    new_value = 0
                    weight_sum = 0

                    # Iterate over local 3d patch for each pixel
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            for k in range(k_start, k_end):
                                weight = patch_distance_3d[np_floats](
                                    central_patch,
                                    padded[i:i+s, j:j+s, k:k+s],
                                    w, s, var)
                                # Collect results in weight sum
                                weight_sum += weight
                                new_value += weight * padded[i+offset,
                                                             j+offset,
                                                             k+offset]

                    # Normalize the result
                    result[pln, row, col] = new_value / weight_sum

    return np.asarray(result)

#-------------- Accelerated algorithm of Froment 2015 ------------------


cdef inline double _integral_to_distance_2d(double [:, ::] integral,
                                            Py_ssize_t row, Py_ssize_t col,
                                            Py_ssize_t offset,
                                            double h2s2) nogil:
    """
    Parameters
    ----------
    integral : ndarray
        The integral image as computed by ``_integral_image_2d``.
    row, col : Py_ssize_t
        Index of the patch's center pixel.
    offset : Py_ssize_t
        The non-local means patch radius.
    h2s2 : float
        Normalization factor related to the image standard deviation and `h`
        parameter.

    Returns
    -------
    distance : float
        The patch distance

    Notes
    -----
    Used in `_fast_nl_means_denoising_2d` which is a fast non-local means
    algorithm using integral images as described in [1]_, [2]_.

    References
    ----------
    .. [1] J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen. Fast
           nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
           International Symposium on Biomedical Imaging: From Nano to Macro,
           2008, pp. 1331-1334.
    .. [2] Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
           Denoising. Image Processing On Line, 2014, vol. 4, pp. 300-326.
    """
    cdef double distance = (integral[row + offset, col + offset] +
                            integral[row - offset, col - offset] -
                            integral[row - offset, col + offset] -
                            integral[row + offset, col - offset])
    return max(distance, 0.0) / h2s2


cdef inline double _integral_to_distance_3d(double[:, :, ::] integral,
                                            Py_ssize_t pln, Py_ssize_t row,
                                            Py_ssize_t col, Py_ssize_t offset,
                                            double s_cube_h_square) nogil:
    """
    Parameters
    ----------
    integral : ndarray
        The integral image as computed by ``_integral_image_3d``.
    pln, row, col : Py_ssize_t
        Index of the patch's center pixel.
    offset : Py_ssize_t
        The non-local means patch radius.
    s_cube_h_square : float
        Normalization factor related to the image standard deviation and `h`
        parameter.

    Returns
    -------
    distance : float
        The patch distance

    Notes
    -----
    Used in `_fast_nl_means_denoising_3d` which is a fast non-local means
    algorithm using integral images as described in [1]_, [2]_.

    References
    ----------
    .. [1] J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen. Fast
           nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
           International Symposium on Biomedical Imaging: From Nano to Macro,
           2008, pp. 1331-1334.
    .. [2] Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
           Denoising. Image Processing On Line, 2014, vol. 4, pp. 300-326.
    """
    cdef double distance = (
        integral[pln + offset, row + offset, col + offset] -
        integral[pln - offset, row - offset, col - offset] +
        integral[pln - offset, row - offset, col + offset] +
        integral[pln - offset, row + offset, col - offset] +
        integral[pln + offset, row - offset, col - offset] -
        integral[pln - offset, row + offset, col + offset] -
        integral[pln + offset, row - offset, col + offset] -
        integral[pln + offset, row + offset, col - offset])
    return max(distance, 0.0) / (s_cube_h_square)


cdef inline double _integral_to_distance_4d(double [:, :, :, ::] integral,
                                            Py_ssize_t time, Py_ssize_t pln,
                                            Py_ssize_t row, Py_ssize_t col,
                                            Py_ssize_t offset,
                                            double s4_h_square) nogil:
    """
    Parameters
    ----------
    integral : ndarray
        The integral image as computed by ``_integral_image_4d``.
    time, pln, row, col : Py_ssize_t
        Index of the patch's center pixel.
    offset : Py_ssize_t
        The non-local means patch radius.
    s4_h_square : float
        Normalization factor related to the image standard deviation and `h`
        parameter.

    Returns
    -------
    distance : float
        The patch distance

    Notes
    -----
    Used in _fast_nl_means_denoising_4d which is a fast non-local means
    algorithm using integral images as described in [1]_, [2]_. The
    coefficients for the terms in the 4D case were determined using Eq. 54 of
    [3]_ as implemented in the ``integral_image_coefficients`` function.

    References
    ----------
    .. [1] J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen. Fast
           nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
           International Symposium on Biomedical Imaging: From Nano to Macro,
           2008, pp. 1331-1334.
    .. [2] Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
           Denoising. Image Processing On Line, 2014, vol. 4, pp. 300-326.
    .. [3] Tapia, E. A note on the computation of high-dimensional integral
           images. Pattern Recognition Letters, 2011, Vol. 32, pp.197-201.
    """
    cdef double distance
    distance = (
        integral[time - offset, pln - offset, row - offset, col - offset] -
        integral[time - offset, pln - offset, row - offset, col + offset] -
        integral[time - offset, pln - offset, row + offset, col - offset] +
        integral[time - offset, pln - offset, row + offset, col + offset] -
        integral[time - offset, pln + offset, row - offset, col - offset] +
        integral[time - offset, pln + offset, row - offset, col + offset] +
        integral[time - offset, pln + offset, row + offset, col - offset] -
        integral[time - offset, pln + offset, row + offset, col + offset] -
        integral[time + offset, pln - offset, row - offset, col - offset] +
        integral[time + offset, pln - offset, row - offset, col + offset] +
        integral[time + offset, pln - offset, row + offset, col - offset] -
        integral[time + offset, pln - offset, row + offset, col + offset] +
        integral[time + offset, pln + offset, row - offset, col - offset] -
        integral[time + offset, pln + offset, row - offset, col + offset] -
        integral[time + offset, pln + offset, row + offset, col - offset] +
        integral[time + offset, pln + offset, row + offset, col + offset])
    return max(distance, 0.0) / s4_h_square


cdef inline void _integral_image_2d(double [:, :, ::] padded,
                                    double [:, ::] integral,
                                    Py_ssize_t t_row, Py_ssize_t t_col,
                                    Py_ssize_t n_row, Py_ssize_t n_col,
                                    Py_ssize_t n_channels,
                                    double var_diff) nogil:
    """ Compute the integral of the squared difference between an image
    ``padded`` and the same image shifted by ``(t_row, t_col)``.

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
        Shift along the column axis (positive).
    n_row : Py_ssize_t
    n_col : Py_ssize_t
    n_channels : Py_ssize_t
    var_diff : double
        The double of the expected noise variance.  If non-zero, this
        is used to reduce the apparent patch distances by the expected
        distance due to the noise.

    Notes
    -----

    The integral computation could be performed using
    ``transform.integral_image``, but this helper function saves memory
    by avoiding copies of ``padded``.
    """
    cdef Py_ssize_t row, col, channel
    cdef Py_ssize_t row_start = max(1, -t_row)
    cdef Py_ssize_t row_end = min(n_row, n_row - t_row)
    cdef double t, distance

    for row in range(row_start, row_end):
        for col in range(1, n_col - t_col):
            distance = 0
            for channel in range(n_channels):
                t = (padded[row, col, channel] -
                     padded[row + t_row, col + t_col, channel])
                distance += t * t
            distance -= n_channels * var_diff
            integral[row, col] = (distance +
                                  integral[row - 1, col] +
                                  integral[row, col - 1] -
                                  integral[row - 1, col - 1])


cdef inline void _integral_image_3d(double [:, :, :, ::] padded,
                                    double [:, :, ::] integral,
                                    Py_ssize_t t_pln, Py_ssize_t t_row,
                                    Py_ssize_t t_col, Py_ssize_t n_pln,
                                    Py_ssize_t n_row, Py_ssize_t n_col,
                                    Py_ssize_t n_channels,
                                    double var_diff) nogil:
    """Compute the integral of the squared difference between an image ``padded``
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
        Shift along the column axis (positive).
    n_pln : Py_ssize_t
    n_row : Py_ssize_t
    n_col : Py_ssize_t
    n_channels : Py_ssize_t
    var_diff : np_floats
        The double of the expected noise variance.  If non-zero, this
        is used to reduce the apparent patch distances by the expected
        distance due to the noise.

    Notes
    -----

    The integral computation could be performed using
    ``transform.integral_image``, but this helper function saves memory
    by avoiding copies of ``padded``.
    """
    cdef Py_ssize_t pln, row, col
    cdef Py_ssize_t pln_start = max(1, -t_pln)
    cdef Py_ssize_t pln_end = min(n_pln, n_pln - t_pln)
    cdef Py_ssize_t row_start = max(1, -t_row)
    cdef Py_ssize_t row_end = min(n_row, n_row - t_row)
    cdef double t, distance

    for pln in range(pln_start, pln_end):
        for row in range(row_start, row_end):
            for col in range(1, n_col - t_col):
                distance = 0
                for channel in range(n_channels):
                    t = (padded[pln, row, col, channel] -
                         padded[pln + t_pln, row + t_row, col + t_col,
                                channel])
                    distance += t * t
                distance -= n_channels * var_diff
                integral[pln, row, col] = (
                    distance +
                    integral[pln - 1, row, col] +
                    integral[pln, row - 1, col] +
                    integral[pln, row, col - 1] +
                    integral[pln - 1, row - 1, col - 1] -
                    integral[pln - 1, row - 1, col] -
                    integral[pln, row - 1, col - 1] -
                    integral[pln - 1, row, col - 1])


cdef inline void _integral_image_4d(double [:, :, :, :, ::] padded,
                                    double [:, :, :, ::] integral,
                                    Py_ssize_t t_time, Py_ssize_t t_pln, Py_ssize_t t_row,
                                    Py_ssize_t t_col, Py_ssize_t n_time, Py_ssize_t n_pln,
                                    Py_ssize_t n_row, Py_ssize_t n_col, Py_ssize_t n_channels,
                                    double var_diff) nogil:
    """Compute the integral of the squared difference between an image ``padded``
    and the same image shifted by ``(t_pln, t_row, t_col)``.

    Parameters
    ----------
    padded : ndarray of shape (n_time, n_pln, n_row, n_col)
        Image of interest.
    integral : ndarray
        Output of the function. The array is filled with integral values.
        ``integral`` should have the same shape as ``padded``.
    t_time : Py_ssize_t
        Shift along the time axis.
    t_pln : Py_ssize_t
        Shift along the plane axis.
    t_row : Py_ssize_t
        Shift along the row axis.
    t_col : Py_ssize_t
        Shift along the column axis.
    n_time : Py_ssize_t
    n_pln : Py_ssize_t
    n_row : Py_ssize_t
    n_col : Py_ssize_t
    var_diff : double
        The double of the expected noise variance. If non-zero, this
        is used to reduce the apparent patch distances by the expected
        distance due to the noise.

    Notes
    -----

    The integral computation could be performed using
    ``transform.integral_image``, but this helper function saves memory
    by avoiding copies of ``padded``.
    """
    cdef Py_ssize_t time, pln, row, col, channel
    cdef Py_ssize_t time_start = max(1, -t_time)
    cdef Py_ssize_t time_end = min(n_time, n_time - t_time)
    cdef Py_ssize_t pln_start = max(1, -t_pln)
    cdef Py_ssize_t pln_end = min(n_pln, n_pln - t_pln)
    cdef Py_ssize_t row_start = max(1, -t_row)
    cdef Py_ssize_t row_end = min(n_row, n_row - t_row)
    cdef double t, distance

    for time in range(time_start, time_end):
        for pln in range(pln_start, pln_end):
            for row in range(row_start, row_end):
                for col in range(1, n_col - t_col):
                    distance = 0
                    for channel in range(n_channels):
                        t = (padded[time, pln, row, col, channel] -
                             padded[time + t_time, pln + t_pln,
                                    row + t_row, col + t_col, channel])
                        distance += t * t
                    distance -= n_channels * var_diff

                    integral[time, pln, row, col] = (
                        distance +
                        # add terms with shift along 1 axis
                        integral[time - 1, pln, row, col] +
                        integral[time, pln - 1, row, col] +
                        integral[time, pln, row - 1, col] +
                        integral[time, pln, row, col - 1] +
                        # add terms with shift along 3 axes
                        integral[time, pln - 1, row - 1, col - 1] +
                        integral[time - 1, pln, row - 1, col - 1] +
                        integral[time - 1, pln - 1, row, col - 1] +
                        integral[time - 1, pln - 1, row - 1, col] -
                        # subtract terms with shift along 2 axes
                        integral[time, pln, row - 1, col - 1] -
                        integral[time, pln - 1, row, col - 1] -
                        integral[time, pln - 1, row - 1, col] -
                        integral[time - 1, pln, row, col - 1] -
                        integral[time - 1, pln, row - 1, col] -
                        integral[time - 1, pln - 1, row, col] -
                        # subtract term with shift along 4 axes
                        integral[time - 1, pln - 1, row - 1, col - 1])


def _fast_nl_means_denoising_2d(cnp.ndarray[np_floats, ndim=3] image,
                                Py_ssize_t s, Py_ssize_t d,
                                double h, double var):
    """Perform fast non-local means denoising on 2-D array, with the outer
    loop on patch shifts in order to reduce the number of operations.

    Parameters
    ----------
    image : ndarray
        2-D input data to be denoised, grayscale or RGB.
    s : Py_ssize_t, optional
        Size of patches used for denoising.
    d : Py_ssize_t, optional
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
    ..[1] J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen, Fast
          nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
          International Symposium on Biomedical Imaging: From Nano to Macro,
          2008, pp. 1331-1334.

    ..[2] Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
          Denoising. Image Processing On Line, 2014, vol. 4, pp. 300-326.
    """

    cdef double DISTANCE_CUTOFF = 5.0
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    # Image padding: we need to account for patch size, possible shift,
    # + 1 for the boundary effects in finite differences
    cdef Py_ssize_t n_row, n_col, t_row, t_col, row, col, n_channels, channel
    cdef Py_ssize_t row_start, row_end, row_shift, col_shift
    cdef Py_ssize_t offset = s / 2
    cdef Py_ssize_t pad_size = offset + d + 1

    cdef double [:, :, ::1] padded = np.ascontiguousarray(
        np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
               mode='reflect').astype(np.float64))
    cdef double [:, ::1] weights = np.zeros_like(padded[..., 0])
    cdef double [:, ::1] integral = np.zeros_like(weights)
    cdef double [:, :, ::1] result = np.zeros_like(padded)
    cdef double distance, h2s2, weight, alpha

    n_row, n_col, n_channels = padded.shape[0], padded.shape[1], padded.shape[2]
    h2s2 = n_channels * h * h * s * s
    var *= 2

    with nogil:
        # Outer loops on patch shifts
        # With t2 >= 0, reference patch is always on the left of test patch
        # Iterate over shifts along the row axis
        for t_row in range(-d, d + 1):
            row_start = max(offset, offset - t_row)
            row_end = min(n_row - offset, n_row - offset - t_row)
            # Iterate over shifts along the column axis
            for t_col in range(0, d + 1):
                # alpha is to account for patches on the same column
                # distance is computed twice in this case
                alpha = 0.5 if t_col == 0 else 1

                # Compute integral image of the squared difference between
                # padded and the same image shifted by (t_row, t_col)
                _integral_image_2d(padded, integral, t_row, t_col,
                                   n_row, n_col, n_channels, var)

                # Inner loops on pixel coordinates
                # Iterate over rows, taking offset and shift into account
                for row in range(row_start, row_end):
                    row_shift = row + t_row
                    # Iterate over columns, taking offset and shift into account
                    for col in range(offset, n_col - offset - t_col):
                        # Compute squared distance between shifted patches
                        distance = _integral_to_distance_2d(
                            integral, row, col, offset, h2s2)
                        # exp of large negative numbers is close to zero
                        if distance > DISTANCE_CUTOFF:
                            continue
                        col_shift = col + t_col
                        weight = alpha * _fast_exp(-distance)
                        # Accumulate weights corresponding to different shifts
                        weights[row, col] += weight
                        weights[row_shift, col_shift] += weight
                        # Iterate over channels
                        for channel in range(n_channels):
                            result[row, col, channel] += weight * \
                                padded[row_shift, col_shift, channel]
                            result[row_shift, col_shift, channel] += \
                                weight * padded[row, col, channel]

        # Normalize pixel values using sum of weights of contributing patches
        for row in range(pad_size, n_row - pad_size):
            for col in range(pad_size, n_col - pad_size):
                for channel in range(n_channels):
                    # No risk of division by zero, since the contribution
                    # of a null shift is strictly positive
                    result[row, col, channel] /= weights[row, col]

    # Return cropped result, undoing padding
    return np.squeeze(np.asarray(result[pad_size: -pad_size,
                                        pad_size: -pad_size, :]).astype(dtype))


def _fast_nl_means_denoising_3d(cnp.ndarray[np_floats, ndim=4] image,
                                Py_ssize_t s=5, Py_ssize_t d=7, double h=0.1,
                                double var=0.):
    """Perform fast non-local means denoising on 3-D array, with the outer
    loop on patch shifts in order to reduce the number of operations.

    Parameters
    ----------
    image : ndarray
        3-D input data to be denoised.
    s : Py_ssize_t, optional
        Size of patches used for denoising.
    d : Py_ssize_t, optional
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
    ..[1] J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen, Fast
          nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
          International Symposium on Biomedical Imaging: From Nano to Macro,
          2008, pp. 1331-1334.

    ..[2] Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
          Denoising. Image Processing On Line, 2014, vol. 4, pp. 300-326.
    """

    cdef double DISTANCE_CUTOFF = 5.0
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef Py_ssize_t offset = s / 2
    # Image padding: we need to account for patch size, possible shift,
    # + 1 for the boundary effects in finite differences
    cdef Py_ssize_t pad_size = offset + d + 1
    cdef double [:, :, :, ::1] padded = np.ascontiguousarray(
        np.pad(image,
               ((pad_size, pad_size),
                (pad_size, pad_size),
                (pad_size, pad_size),
                (0, 0)),
               mode='reflect'),
        dtype=np.float64)
    cdef double [:, :, ::1] weights = np.zeros_like(padded[..., 0])
    cdef double [:, :, ::1] integral = np.zeros_like(padded[..., 0])
    cdef double [:, :, :, ::1] result = np.zeros_like(padded)

    cdef Py_ssize_t n_pln, n_row, n_col, t_pln, t_row, t_col, \
             pln, row, col, channel, n_channels
    cdef Py_ssize_t pln_dist_min, pln_dist_max, row_dist_min, row_dist_max, \
             col_dist_min, col_dist_max
    cdef double weight, distance, alpha
    n_pln, n_row, n_col, n_channels = padded.shape[0], padded.shape[1], padded.shape[2], padded.shape[3]
    cdef double s_cube_h_square = n_channels * h * h * s * s * s


    var *= 2

    with nogil:
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
                    # alpha is to account for patches on the same column
                    # distance is computed twice in this case
                    alpha = 0.5 if t_col == 0 else 1

                    col_dist_min = offset
                    col_dist_max = n_col - offset - t_col

                    # Compute integral image of the squared difference between
                    # padded and the same image shifted by (t_pln, t_row, t_col)
                    _integral_image_3d(padded, integral, t_pln,
                                       t_row, t_col, n_pln, n_row,
                                       n_col, n_channels, var)

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
                                distance = _integral_to_distance_3d(integral,
                                    pln, row, col, offset, s_cube_h_square)
                                # exp of large negative numbers is close to zero
                                if distance > DISTANCE_CUTOFF:
                                    continue

                                weight = alpha * _fast_exp(-distance)
                                # Accumulate weights for the different shifts
                                weights[pln, row, col] += weight
                                weights[pln + t_pln, row + t_row,
                                                     col + t_col] += weight
                                for channel in range(n_channels):
                                    result[pln, row, col, channel] += weight * \
                                            padded[pln + t_pln, row + t_row,
                                                   col + t_col, channel]
                                    result[pln + t_pln, row + t_row,
                                           col + t_col, channel] += weight * \
                                                                    padded[pln, row, col, channel]

        # Normalize pixel values using sum of weights of contributing patches
        for pln in range(offset, n_pln - offset):
            for row in range(offset, n_row - offset):
                for col in range(offset, n_col - offset):
                    for channel in range(n_channels):
                        # No risk of division by zero, since the contribution
                        # of a null shift is strictly positive
                        result[pln, row, col, channel] /= weights[pln, row, col]

    # Return cropped result, undoing padding
    return np.squeeze(
        np.asarray(result[pad_size:-pad_size,
                          pad_size:-pad_size,
                          pad_size:-pad_size, :], dtype=dtype)
    )


def _fast_nl_means_denoising_4d(cnp.ndarray[np_floats, ndim=5] image,
                                Py_ssize_t s=3, Py_ssize_t d=3, double h=0.1,
                                double var=0.):
    """
    Perform fast non-local means denoising on 3-D array, with the outer
    loop on patch shifts in order to reduce the number of operations.

    Parameters
    ----------
    image : ndarray
        4-D input data to be denoised.
    s : Py_ssize_t, optional
        Size of patches used for denoising.
    d : tuple of Py_ssize_t, optional
        Maximal distance in pixels along each axis to search for patches used for denoising.
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
    ..[1] J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen, Fast
          nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
          International Symposium on Biomedical Imaging: From Nano to Macro,
          2008, pp. 1331-1334.

    ..[2] Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
          Denoising. Image Processing On Line, 2014, vol. 4, pp. 300-326.
    """

    cdef double DISTANCE_CUTOFF = 5.0
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef Py_ssize_t offset = s / 2
    # Image padding: we need to account for patch size, possible shift,
    # + 1 for the boundary effects in finite differences
    cdef Py_ssize_t pad_size = offset + d + 1
    cdef double [:, :, :, :, ::1] padded = np.ascontiguousarray(
        np.pad(image,
               ((pad_size, pad_size),
                (pad_size, pad_size),
                (pad_size, pad_size),
                (pad_size, pad_size),
                (0, 0)),
               mode='reflect'),
        dtype=np.float64)
    cdef double [:, :, :, :, ::1] result = np.zeros_like(padded)
    cdef double [:, :, :, ::1] weights = np.zeros_like(padded[..., 0])
    cdef double [:, :, :, ::1] integral = np.zeros_like(padded[..., 0])
    cdef Py_ssize_t n_pln, n_row, n_col, t_pln, t_row, t_col, \
             pln, row, col, channel, n_channels, t_time, n_time, time
    cdef Py_ssize_t time_dist_min, time_dist_max, pln_dist_min, pln_dist_max, \
             row_dist_min, row_dist_max, col_dist_min, col_dist_max,
    cdef Py_ssize_t d_row, d_col, d_pln, d_time
    cdef double weight, distance, alpha
    n_time, n_pln, n_row, n_col, n_channels = padded.shape[0], padded.shape[1], padded.shape[2], padded.shape[3], padded.shape[4]
    cdef double s4_h_square = n_channels * h * h * s * s * s * s

    # Outer loops on patch shifts
    # With t2 >= 0, reference patch is always on the left of test patch
    # Iterate over shifts along the plane axis
    var *= 2
    with nogil:
        for t_time in range(-d, d + 1):
            time_dist_min = max(offset, offset - t_time)
            time_dist_max = min(n_time - offset, n_time - offset - t_time)
            for t_pln in range(-d, d + 1):
                pln_dist_min = max(offset, offset - t_pln)
                pln_dist_max = min(n_pln - offset, n_pln - offset - t_pln)
                # Iterate over shifts along the row axis
                for t_row in range(-d, d + 1):
                    row_dist_min = max(offset, offset - t_row)
                    row_dist_max = min(n_row - offset, n_row - offset - t_row)
                    # Iterate over shifts along the column axis
                    for t_col in range(0, d + 1):
                        # alpha is to account for patches on the same column
                        # distance is computed twice in this case
                        alpha = 0.5 if t_col == 0 else 1

                        col_dist_min = offset
                        col_dist_max = n_col - offset - t_col

                        # Compute integral image of the squared difference between
                        # padded and the same image shifted by (t_pln, t_row, t_col)
                        _integral_image_4d(padded, integral, t_time, t_pln, t_row,
                                           t_col, n_time, n_pln, n_row, n_col,
                                           n_channels, var)

                        # Inner loops on pixel coordinates
                        # Iterate over planes, taking offset and shift into account
                        for time in range(time_dist_min, time_dist_max):
                            for pln in range(pln_dist_min, pln_dist_max):
                                # Iterate over rows, taking offset and shift
                                # into account
                                for row in range(row_dist_min, row_dist_max):
                                    # Iterate over columns
                                    for col in range(col_dist_min, col_dist_max):
                                        # Compute squared distance between
                                        # shifted patches
                                        distance = _integral_to_distance_4d(
                                            integral, time, pln, row, col, offset,
                                            s4_h_square)
                                        # exp of large negative numbers is close to zero
                                        if distance > DISTANCE_CUTOFF:
                                            continue

                                        weight = alpha * _fast_exp(-distance)
                                        # Accumulate weights for the different shifts
                                        weights[time, pln, row, col] += weight
                                        weights[time + t_time, pln + t_pln,
                                                row + t_row, col + t_col] += weight
                                        for channel in range(n_channels):
                                            result[time, pln, row, col,
                                                   channel] += weight * \
                                                       padded[time + t_time,
                                                              pln + t_pln,
                                                              row + t_row,
                                                              col + t_col, channel]
                                            result[time + t_time, pln + t_pln,
                                                   row + t_row, col + t_col,
                                                   channel] += weight * \
                                                       padded[time, pln, row,
                                                              col, channel]

        # Normalize pixel values using sum of weights of contributing patches
        for time in range(offset, n_time - offset):
            for pln in range(offset, n_pln - offset):
                for row in range(offset, n_row - offset):
                    for col in range(offset, n_col - offset):
                        for channel in range(n_channels):
                            # No risk of division by zero, since the contribution
                            # of a null shift is strictly positive
                            result[time, pln, row, col, channel] /= weights[
                                time, pln, row, col]

    # Return cropped result, undoing padding
    return np.squeeze(
        np.asarray(result[pad_size:-pad_size,
                          pad_size:-pad_size,
                          pad_size:-pad_size,
                          pad_size:-pad_size, :], dtype=dtype)
    )
