import numpy as np
cimport cython
from libc.math cimport isnan, INFINITY

from .._shared.fused_numerics cimport np_floats

ctypedef np_floats DTYPE_FLOAT

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_kernel_nan(DTYPE_FLOAT[:,:,:,:] windows,
                     DTYPE_FLOAT[:,:] kernel,
                     DTYPE_FLOAT[:,:] cap_height):
    
    cdef DTYPE_FLOAT[:, ::1] out_data = np.zeros((windows.shape[0], windows.shape[1]), dtype=windows.base.dtype)
    cdef int im_x, im_y, kern_x, kern_y
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
    
    cdef DTYPE_FLOAT[:, :] out_data = np.zeros((windows.shape[0], windows.shape[1]), dtype=windows.base.dtype)
    cdef int im_x, im_y, kern_x, kern_y
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
