# -*- python -*-

cimport numpy as np
import numpy as np

def possible_hull(np.ndarray[dtype=np.uint8_t, ndim=2, mode="c"] img):
    """Return positions of pixels that possibly belong to the convex hull.

    Parameters
    ----------
    img : ndarray of bool
        Binary input image.

    Returns
    -------
    coords : ndarray (N, 2)
       The ``(row, column)`` coordinates of all pixels that possibly belong to
       the convex hull.

    """
    cdef int i, j, k
    cdef unsigned int M, N
    
    M = img.shape[0]
    N = img.shape[1]

    # Output: M storage slots for left boundary pixels
    #         N storage slots for top boundary pixels
    #         M storage slots for right boundary pixels
    #         N storage slots for bottom boundary pixels
    cdef np.ndarray[dtype=np.int_t, ndim=2] nonzero = \
         np.ones((2 * (M + N), 2), dtype=np.int)
    nonzero *= -1 

    k = 0
    for i in range(M):
        for j in range(N):
            if img[i, j] != 0:
                # Left check
                if nonzero[i, 1] == -1:
                    nonzero[i, 0] = i
                    nonzero[i, 1] = j

                # Right check
                elif nonzero[M + N + i, 1] < j:
                    nonzero[M + N + i, 0] = i
                    nonzero[M + N + i, 1] = j

                # Top check
                if nonzero[M + j, 1] == -1:
                    nonzero[M + j, 0] = i
                    nonzero[M + j, 1] = j

                # Bottom check
                elif nonzero[2 * M + N + j, 0] < i:
                    nonzero[2 * M + N + j, 0] = i
                    nonzero[2 * M + N + j, 1] = j
    
    return nonzero[nonzero[:, 0] != -1]
