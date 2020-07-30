import numpy as np
cimport numpy as np

cimport cython

ctypedef np.float_t DTYPE

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_kernel(np.ndarray[DTYPE, ndim=4] windows,
                 np.ndarray[DTYPE, ndim=2] kernel,
                 np.ndarray[DTYPE, ndim=2] sagitta):
    cdef np.ndarray out_array = np.zeros((windows.shape[0], windows.shape[1]))
    cdef DTYPE[:, :] out_data = out_array
    cdef int kernel_size = kernel.size
    cdef np.ndarray[DTYPE, ndim=1] tmp_array = np.empty(kernel_size)
    cdef DTYPE[:] tmp = tmp_array
    cdef int idx, tmp_idx, im_x, im_y, kern_x, kern_y
    cdef DTYPE min_value

    for im_y in range(windows.shape[0]):
        for im_x in range(windows.shape[1]):
            for kern_y in range(kernel.shape[0]):
                for kern_x in range(kernel.shape[1]):
                    tmp_idx = kern_y * kernel.shape[1] + kern_x
                    tmp[tmp_idx] = (windows[im_y, im_x, kern_y, kern_x] + sagitta[kern_y, kern_x]) * kernel[kern_y, kern_x]
            
            min_value = tmp[0]
            for idx in range(kernel_size):
                if min_value > tmp[idx]:
                    min_value = tmp[idx]
            out_data[im_y, im_x] = min_value

    return out_array