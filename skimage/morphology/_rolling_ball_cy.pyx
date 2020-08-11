import numpy as np
cimport cython
from libc.math cimport isnan, INFINITY

from .._shared.fused_numerics cimport np_floats

ctypedef np_floats DTYPE_FLOAT

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_kernel_nan(DTYPE_FLOAT[:,:,:,:] windows,
                     DTYPE_FLOAT[:,::1] kernel,
                     DTYPE_FLOAT[:,::1] cap_height):
    """
    apply_kernel_nan(windows, kernel, cap_height)

    Apply a custom kernel to a windowed view of an image.

    This function is the critical piece of code for 
    `morphology.rolling_ellipsoid`. It was moved to cython for speed.

    Parameters
    ----------
    windows : (N, M, K1, K2) ndarray
        A windowed view into a 2D image.
    kernel : (K1, K2) ndarray
        Indicates if pixel inside the window belongs to the kernel. 
        `kernel[x,y] == 1` if the pixel is inside, ``kernel[x,y] == np.Inf`` 
        otherwise.
    cap_height : (K1, K2) ndarray
        Indicates the height/intensity of the ellipsoid at position ``(x,y)``

    Returns
    -------
    out_data : (N, M) ndarray
        2D Image estimating the background intensity.

    See Also
    --------
    rolling_ellipsoid
    """
    
    cdef DTYPE_FLOAT[:, ::1] out_data = np.zeros((windows.shape[0], windows.shape[1]), dtype=windows.base.dtype)
    cdef Py_ssize_t im_x, im_y, kern_x, kern_y
    cdef DTYPE_FLOAT min_value, tmp

    with nogil:
        for im_y in range(windows.shape[0]):
            for im_x in range(windows.shape[1]):
                min_value = INFINITY
                for kern_y in range(kernel.shape[0]):
                    for kern_x in range(kernel.shape[1]):
                        tmp = (windows[im_y, im_x, kern_y, kern_x] + cap_height[kern_y, kern_x]) * kernel[kern_y, kern_x]
                        if not min_value <= tmp:
                            min_value = tmp
                            if isnan(tmp):
                                break
                    if isnan(min_value):
                        break
                out_data[im_y, im_x] = min_value

    return out_data.base


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_kernel(DTYPE_FLOAT[:,:,:,:] windows,
                 DTYPE_FLOAT[:, ::1] kernel,
                 DTYPE_FLOAT[:, ::1] cap_height):
    """
    apply_kernel(windows, kernel, cap_height)

    Apply a custom kernel to a windowed view of an image.

    This function is the critical piece of code for 
    `morphology.rolling_ellipsoid`. It was moved to cython for speed.

    This function assumes that the image doesn't contain ``NaN``s; this 
    assumption allows for faster code (better compiler optimization).

    Parameters
    ----------
    windows : (N, M, K1, K2) ndarray
        A windowed view into a 2D image.
    kernel : (K1, K2) ndarray
        Indicates if pixel inside the window belongs to the kernel. 
        `kernel[x,y] == 1` if the pixel is inside, ``kernel[x,y] == np.Inf`` 
        otherwise.
    cap_height : (K1, K2) ndarray
        Indicates the height/intensity of the ellipsoid at position ``(x,y)``

    Returns
    -------
    out_data : (N, M) ndarray
        2D Image estimating the background intensity.

    See Also
    --------
    rolling_ellipsoid
    """
    
    cdef DTYPE_FLOAT[:, ::1] out_data = np.zeros((windows.shape[0], windows.shape[1]), dtype=windows.base.dtype)
    cdef Py_ssize_t im_x, im_y, kern_x, kern_y
    cdef DTYPE_FLOAT min_value, tmp

    with nogil:
        for im_y in range(windows.shape[0]):
            for im_x in range(windows.shape[1]):
                min_value = INFINITY
                for kern_y in range(kernel.shape[0]):
                    for kern_x in range(kernel.shape[1]):
                        tmp = (windows[im_y, im_x, kern_y, kern_x] + cap_height[kern_y, kern_x]) * kernel[kern_y, kern_x]
                        if min_value > tmp:
                            min_value = tmp
                out_data[im_y, im_x] = min_value

    return out_data.base

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_kernel_flat(DTYPE_FLOAT[:,:,:] windows,
                 DTYPE_FLOAT[:, ::1] kernel,
                 DTYPE_FLOAT[:, ::1] cap_height):
    """
    apply_kernel_flat(windows, kernel, cap_height)

    Apply a custom kernel to a windowed view of an image.

    This function is the critical piece of code for 
    `morphology.rolling_ellipsoid`. It was moved to cython for speed.

    This function assumes that the image doesn't contain ``NaN``s; this 
    assumption allows for faster code (better compiler optimization).

    Parameters
    ----------
    windows : (N, K1, K2) ndarray
        A windowed view into an nD image. K1 and K2 are the dimensions of the
        window and N is the number of windows into the image.
    kernel : (K1, K2) ndarray
        Indicates if pixel inside the window belongs to the kernel. 
        `kernel[x,y] == 1` if the pixel is inside, ``kernel[x,y] == np.Inf`` 
        otherwise.
    cap_height : (K1, K2) ndarray
        Indicates the height/intensity of the ellipsoid at position ``(x,y)``

    Returns
    -------
    out_data : (N, M) ndarray
        2D Image estimating the background intensity.

    See Also
    --------
    rolling_ellipsoid

    Notes
    -----
    The flattened version can handle arbitrary image dimensions by flattening
    all image dimensions and then inflating it after computation. Copies can 
    be avoided using two views; one before and after the function call.
    For an example refer to the implementation of ``rolling_ellipsoid``.
    """
    
    cdef DTYPE_FLOAT[::1] out_data = np.zeros(windows.shape[0], dtype=windows.base.dtype)
    cdef Py_ssize_t offset, kern_x, kern_y
    cdef DTYPE_FLOAT min_value, tmp

    with nogil:
        for offset in range(windows.shape[0]):
            min_value = INFINITY
            for kern_y in range(kernel.shape[0]):
                for kern_x in range(kernel.shape[1]):
                    tmp = (windows[offset, kern_y, kern_x] + cap_height[kern_y, kern_x]) * kernel[kern_y, kern_x]
                    if min_value > tmp:
                        min_value = tmp
            out_data[offset] = min_value

    return out_data.base