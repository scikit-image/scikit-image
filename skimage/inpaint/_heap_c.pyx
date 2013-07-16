import heapq
cimport numpy as cnp
import numpy as np
from skimage.morphology import dilation, disk

cdef:
    unsigned int KNOWN = 0
    unsigned int BAND = 1
    unsigned int INSIDE = 2
# KNOWN = 0
# BAND = 1
# INSIDE = 2

def _initialise(cnp.ndarray[cnp.uint8_t, ndim=2] _mask):
    cdef:
        cnp.ndarray[cnp.uint8_t, ndim=2] outside
        cnp.ndarray[cnp.uint8_t, ndim=2] band
        cnp.ndarray[cnp.uint8_t, ndim=2] flag
        cnp.ndarray[double, ndim=2] u
        list heap = list()
        cnp.ndarray[cnp.uint8_t, ndim=2] indices
        cnp.ndarray[cnp.uint8_t, ndim=1] z

    mask = np.ascontiguousarray(_mask)
    outside = dilation(mask, disk(1))
    band = np.logical_xor(mask, outside).astype(np.uint8)

    flag =  (2 * outside) - band
    u = np.where(flag == INSIDE, 1.0e6, 0)

    indices = np.transpose(np.where(flag == BAND)).astype(np.uint8)
    for z in indices:
        heapq.heappush(heap, (u[tuple(z)], tuple(z)))

    return flag, u, heap
