#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp
cnp.import_array()


def possible_hull(cnp.uint8_t[:, ::1] img):
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
    coords = np.ones((2 * (rows + cols), 2), dtype=np.intp)
    coords *= -1

    cdef Py_ssize_t[:, ::1] nonzero = coords
    cdef Py_ssize_t rows_cols = rows + cols
    cdef Py_ssize_t rows_2_cols = 2 * rows + cols
    cdef Py_ssize_t rows_cols_r, rows_c

    with nogil:
        for r in range(rows):

            rows_cols_r = rows_cols + r

            for c in range(cols):

                if img[r, c] != 0:

                    rows_c = rows + c
                    rows_2_cols_c = rows_2_cols + c

                    # Left check
                    if nonzero[r, 1] == -1:
                        nonzero[r, 0] = r
                        nonzero[r, 1] = c

                    # Right check
                    elif nonzero[rows_cols_r, 1] < c:
                        nonzero[rows_cols_r, 0] = r
                        nonzero[rows_cols_r, 1] = c

                    # Top check
                    if nonzero[rows_c, 1] == -1:
                        nonzero[rows_c, 0] = r
                        nonzero[rows_c, 1] = c
  
                    # Bottom check
                    elif nonzero[rows_2_cols_c, 0] < r:
                        nonzero[rows_2_cols_c, 0] = r
                        nonzero[rows_2_cols_c, 1] = c

    return coords[coords[:, 0] != -1]
