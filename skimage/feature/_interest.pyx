#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libc.float cimport DBL_MAX


def moravec(cnp.ndarray[dtype=cnp.double_t, ndim=2, mode='c'] image,
             int block_size):
    cdef int rows = image.shape[0]
    cdef int cols = image.shape[1]

    cdef cnp.ndarray[dtype=cnp.double_t, ndim=2, mode='c'] out = \
         np.zeros_like(image)

    cdef double* image_data = <double*>image.data
    cdef double* out_data = <double*>out.data

    cdef double msum, min_msum
    cdef int r, c, br, bc, mr, mc, a, b
    for r in range(2 * block_size, rows - 2 * block_size):
        for c in range(2 * block_size, cols - 2 * block_size):
            min_msum = DBL_MAX
            for br in range(r - block_size, r + block_size + 1):
                for bc in range(c - block_size, c + block_size + 1):
                    if br != r and bc != c:
                        msum = 0
                        for mr in range(- block_size, block_size + 1):
                            for mc in range(- block_size, block_size + 1):
                                a = (r + mr) * cols + c + mc
                                b = (br + mr) * cols + bc + mc
                                msum += (image_data[a] - image_data[b]) ** 2
                        min_msum = min(msum, min_msum)

            out_data[r * cols + c] = min_msum

    return out