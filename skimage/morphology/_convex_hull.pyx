#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp


def possible_hull(cnp.ndarray[dtype=cnp.uint8_t, ndim=2, mode="c"] img):
    """Return positions of pixels that possibly belong to the convex hull.

    Parameters
    ----------
    img : ndarray of bool
        Binary input image.

    Returns
    -------
    coords : ndarray (cols, 2)
       The ``(row, column)`` coordinates of all pixels that possibly belong to
       the convex hull.

    """
    cdef Py_ssize_t r, c
    cdef Py_ssize_t rows = img.shape[0]
    cdef Py_ssize_t cols = img.shape[1]

    # Output: rows storage slots for left boundary pixels
    #         cols storage slots for top boundary pixels
    #         rows storage slots for right boundary pixels
    #         cols storage slots for bottom boundary pixels
    cdef cnp.ndarray[dtype=cnp.intp_t, ndim=2] nonzero = \
         np.ones((2 * (rows + cols), 2), dtype=np.intp)
    nonzero *= -1

    for r in range(rows):
        for c in range(cols):
            if img[r, c] != 0:
                # Left check
                if nonzero[r, 1] == -1:
                    nonzero[r, 0] = r
                    nonzero[r, 1] = c

                # Right check
                elif nonzero[rows + cols + r, 1] < c:
                    nonzero[rows + cols + r, 0] = r
                    nonzero[rows + cols + r, 1] = c

                # Top check
                if nonzero[rows + c, 1] == -1:
                    nonzero[rows + c, 0] = r
                    nonzero[rows + c, 1] = c

                # Bottom check
                elif nonzero[2 * rows + cols + c, 0] < r:
                    nonzero[2 * rows + cols + c, 0] = r
                    nonzero[2 * rows + cols + c, 1] = c

    return nonzero[nonzero[:, 0] != -1]
