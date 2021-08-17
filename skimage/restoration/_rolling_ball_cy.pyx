import numpy as np
cimport cython
from libc.math cimport isnan, INFINITY
from cython.parallel cimport prange

from .._shared.fused_numerics cimport np_floats

ctypedef np_floats DTYPE_FLOAT

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline Py_ssize_t ind2ind(
    Py_ssize_t from_index,
    Py_ssize_t offset,
    Py_ssize_t[::1] from_shape,
    Py_ssize_t[::1] to_shape) nogil:
    """Convert the flat index of one array to a flat index of another array.

    The primary use case for this is if one array is a view into the other
    array and the dimensionality is unknown at compile time (and, hence, typed
    memoryviews can't be used).

    This is the cython equivalent of
    .. code-block:: python

        to_index = np.ravel_multi_index(
            np.unravel_index(from_index, shape=from_shape),
            shape=to_shape
        ) + offset

    However, it doesn't perform boundary checks or type checks; if missused this
    can cause a segfault.

    Parameters
    ----------
    from_index : intp
        The index in the original array.
    offset : intp
        The distance (in number of elements) between the origins
        of both arrays measured from the first element of
        ``from_shape``.
    from_shape : (N) ndarray
        The shape of the original array.
    to_shape : (N) ndarray
        The shape of the resulting array.

    Returns
    -------
    to_index : intp
        The index in the resulting array.
    """

    cdef Py_ssize_t ndim = from_shape.shape[0]
    cdef Py_ssize_t idx, modulo, pos

    cdef Py_ssize_t multiple_kernel = 1
    cdef Py_ssize_t multiple_img = 1
    cdef Py_ssize_t to_index = offset

    for idx in range(ndim-1, -1, -1):
        modulo = from_index % (from_shape[idx] * multiple_kernel)
        pos = modulo // multiple_kernel
        multiple_kernel *= from_shape[idx]

        to_index += pos * multiple_img
        multiple_img *= to_shape[idx]

    return to_index


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_kernel_nan(DTYPE_FLOAT[::1] img not None,
                     DTYPE_FLOAT[::1] intensity_difference not None,
                     DTYPE_FLOAT[::1] out not None,
                     Py_ssize_t[::1] img_shape not None,
                     Py_ssize_t[::1] padded_img_shape not None,
                     Py_ssize_t[::1] kernel_shape not None,
                     Py_ssize_t num_threads=0):
    """Apply a ND kernel to an ND image.

    This function is the critical piece of code for
    `morphology.rolling_ellipsoid`. It was moved to cython for speed.

    Parameters
    ----------
    img : (I) ndarray
        A flat view into a padded image, e.g., from ``numpy.reshape(-1)``.
    intensity_difference : (K) ndarray
        A flat view into an array containing the intensity difference
        between ellipsoid[center] and ellipsoid[pos], e.g., from
        ``numpy.reshape(-1)``. Indicates the difference between the
        height/intensity of the ellipsoid at position ``(x,y)`` and
        the height/intensity at the center of the kernel.
    out : (I) ndarray
        A flat view into the output image. Note: does NOT support inplace
        modification of img.
    img_shape : (N) ndarray
        The shape of the unflattened, unpadded image.
    padded_img_shape : (N) ndarray
        The shape of the unflattened, padded image.
    kernel_shape : (N) ndarray
        The shape of the unflattened kernel.
    num_threads : int, optional
        The number of threads used to compute the result. If no value is
        provided (0, default) fall back to the number of threads that openMP
        is currently configured to use.

    Returns
    -------
    out_data : ndarray
        The array passed into ``out``, reshaped to
        ``out_data.shape = img_shape`` (possibly a view) and filled with the
        estimated background intensity.

    See Also
    --------
    rolling_ball
    """

    cdef Py_ssize_t offset, offset_idx, out_data_size, img_idx
    cdef DTYPE_FLOAT min_value, tmp
    out_data_size = out.size

    cdef Py_ssize_t ndim = kernel_shape.shape[0]
    cdef Py_ssize_t kernel_leading_dims = np.prod(kernel_shape[0:(ndim - 1)])
    cdef Py_ssize_t kernel_last_dim = kernel_shape[ndim - 1]
    cdef Py_ssize_t ker_idx_leading, ker_idx_last

    for offset_idx in prange(
            out_data_size,
            num_threads=num_threads,
            nogil=True):
        offset = ind2ind(offset_idx, 0, img_shape, padded_img_shape)
        min_value = INFINITY

        for ker_idx_leading in range(kernel_leading_dims):
            ker_idx_leading = ker_idx_leading * kernel_last_dim
            img_idx = ind2ind(ker_idx_leading, offset, kernel_shape,
                              padded_img_shape)

            # split into outer and inner loop for vectorization
            # (the inner loop is contiguous in memory)
            for ker_idx_last in range(kernel_last_dim):
                tmp = (img[img_idx+ker_idx_last] +
                       intensity_difference[ker_idx_leading + ker_idx_last])
                if min_value > tmp or isnan(tmp):
                    min_value = tmp
            if isnan(min_value):
                break
        out[offset_idx] = min_value

    return out.base.reshape(img_shape)

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_kernel(DTYPE_FLOAT[::1] img not None,
                 DTYPE_FLOAT[::1] intensity_difference not None,
                 DTYPE_FLOAT[::1] out not None,
                 Py_ssize_t[::1] img_shape not None,
                 Py_ssize_t[::1] padded_img_shape not None,
                 Py_ssize_t[::1] kernel_shape not None,
                 Py_ssize_t num_threads=0):
    """Apply a ND kernel to an ND image.

    This function is the critical piece of code for
    `morphology.rolling_ellipsoid`. It was moved to cython for speed.

    Parameters
    ----------
    img : (I) ndarray
        A flat view into a padded image, e.g., from ``numpy.reshape(-1)``.
    intensity_difference : (K) ndarray
        A flat view into an array containing the intensity difference
        between ellipsoid[center] and ellipsoid[pos], e.g., from
        ``numpy.reshape(-1)``. Indicates the difference between the
        height/intensity of the ellipsoid at position ``(x,y)`` and
        the height/intensity at the center of the kernel.
    out : (I) ndarray
        A flat view into the output image. Note: does NOT support inplace
        modification of img.
    img_shape : (N) ndarray
        The shape of the unflattened, unpadded image.
    padded_img_shape : (N) ndarray
        The shape of the unflattened, padded image.
    kernel_shape : (N) ndarray
        The shape of the unflattened kernel.
    num_threads : int, optional
        The number of threads used to compute the result. If no value is
        provided (0, default) fall back to the number of threads that openMP
        is currently configured to use.

    Returns
    -------
    out_data : ndarray
        The array passed into ``out``, reshaped to
        ``out_data.shape = img_shape`` (possibly a view) and filled with the
        estimated background intensity.

    See Also
    --------
    rolling_ball

    Notes
    -----
    This function assumes that the image doesn't contain ``NaN``s; this
    assumption allows for faster code (better compiler optimization).
    """

    cdef Py_ssize_t offset, offset_idx, out_data_size, ker_idx, img_idx
    cdef DTYPE_FLOAT min_value, tmp
    out_data_size = out.size

    cdef Py_ssize_t ndim = kernel_shape.shape[0]
    cdef Py_ssize_t kernel_leading_dims = np.prod(kernel_shape[0:(ndim - 1)])
    cdef Py_ssize_t kernel_last_dim = kernel_shape[ndim - 1]
    cdef Py_ssize_t ker_idx_leading, ker_idx_last

    for offset_idx in prange(
            out_data_size,
            num_threads=num_threads,
            nogil=True):
        offset = ind2ind(offset_idx, 0, img_shape, padded_img_shape)
        min_value = INFINITY

        # split into outer and inner loop for vectorization
        # (the inner loop is contiguous in memory)
        for ker_idx_leading in range(kernel_leading_dims):
            ker_idx_leading = ker_idx_leading * kernel_last_dim
            img_idx = ind2ind(ker_idx_leading, offset, kernel_shape,
                              padded_img_shape)
            for ker_idx_last in range(kernel_last_dim):
                ker_idx = ker_idx_leading + ker_idx_last
                tmp = (img[img_idx+ker_idx_last] + intensity_difference[ker_idx])
                if min_value > tmp:
                    min_value = tmp
        out[offset_idx] = min_value

    return out.base.reshape(img_shape)
