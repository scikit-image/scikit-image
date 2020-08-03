import numpy as np
cimport numpy as np
cimport cython
from scipy.linalg cimport cython_blas as blas


ctypedef np.double_t DTYPE

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_kernel(DTYPE[:,:,:,:] windows,
                     DTYPE[:,:] kernel,
                     DTYPE[:,:] cap_height,
                     DTYPE max_value=255):
    
    cdef DTYPE[:, :] out_data = np.zeros((windows.shape[0], windows.shape[1]))
    cdef int im_x, im_y, kern_x, kern_y
    cdef DTYPE min_value, tmp

    for im_y in range(windows.shape[0]):
        for im_x in range(windows.shape[1]):
            min_value = max_value
            for kern_y in range(kernel.shape[0]):
                for kern_x in range(kernel.shape[1]):
                    tmp = (windows[im_y, im_x, kern_y, kern_x] + cap_height[kern_y, kern_x]) * kernel[kern_y, kern_x]
                    if min_value > tmp:
                        min_value = tmp
            out_data[im_y, im_x] = min_value

    return out_data.base
