import numpy as np
cimport cython
from libc.math cimport isnan, INFINITY
from cython.parallel cimport prange

from .._shared.fused_numerics cimport np_floats

ctypedef np_floats DTYPE_FLOAT

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_kernel_nan(DTYPE_FLOAT[:,:,:] windows,
                     DTYPE_FLOAT[:,::1] kernel,
                     DTYPE_FLOAT[:,::1] cap_height,
                     Py_ssize_t[::1] offsets):
    """
    apply_kernel_nan(windows, kernel, cap_height)

    Apply a custom kernel to a windowed view of an image.

    This function is the critical piece of code for 
    `morphology.rolling_ellipsoid`. It was moved to cython for speed.

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
    offsets : (N) ndarray
        Array of positions indicating which windows the convolution should be
        applied to

    Returns
    -------
    out_data : (N) ndarray
        1D Array holding the result of the kernel being applied to the windows

    See Also
    --------
    rolling_ellipsoid
    """
    
    cdef DTYPE_FLOAT[::1] out_data = np.zeros(offsets.size, dtype=windows.base.dtype)
    cdef Py_ssize_t offset, offset_idx, offsets_size, kern_x, kern_y
    cdef DTYPE_FLOAT min_value, tmp
    offsets_size = offsets.size

    for offset_idx in prange(offsets_size, nogil=True):
        offset = offsets[offset_idx]
        min_value = INFINITY
        for kern_y in range(kernel.shape[0]):
            for kern_x in range(kernel.shape[1]):
                tmp = (windows[offset, kern_y, kern_x] +
                       cap_height[kern_y, kern_x]) * kernel[kern_y, kern_x]
                if not min_value <= tmp:
                    min_value = tmp
                    if isnan(tmp):
                        break
            if isnan(min_value):
                break
        out_data[offset_idx] = min_value

    return out_data.base

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_kernel(DTYPE_FLOAT[:,:,:] windows,
                 DTYPE_FLOAT[:, ::1] kernel,
                 DTYPE_FLOAT[:, ::1] cap_height,
                 Py_ssize_t[::1] offsets):
    """
    apply_kernel_flat(windows, kernel, cap_height)

    Apply a custom kernel to a windowed view of an image.

    This function is the critical piece of code for 
    `morphology.rolling_ellipsoid`. It was moved to cython for speed.

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
    offsets : (N) ndarray
        1D Array of positions indicating which windows the convolution should
        be applied to

    Returns
    -------
    out_data : (N) ndarray
        1D Array holding the result of the kernel being applied to the windows

    See Also
    --------
    rolling_ellipsoid

    Notes
    -----
    - This function assumes that the image doesn't contain ``NaN``s; this 
    assumption allows for faster code (better compiler optimization).
    """
    
    cdef DTYPE_FLOAT[::1] out_data = np.zeros(offsets.size, dtype=windows.base.dtype)
    cdef Py_ssize_t offset, offset_idx, offsets_size, kern_x, kern_y
    cdef DTYPE_FLOAT min_value, tmp
    offsets_size = offsets.size

    for offset_idx in prange(offsets_size, nogil=True):
        offset = offsets[offset_idx]
        min_value = INFINITY
        for kern_y in range(kernel.shape[0]):
            for kern_x in range(kernel.shape[1]):
                tmp = (windows[offset, kern_y, kern_x] +
                       cap_height[kern_y, kern_x]) * kernel[kern_y, kern_x]
                if min_value > tmp:
                    min_value = tmp
        out_data[offset_idx] = min_value

    return out_data.base