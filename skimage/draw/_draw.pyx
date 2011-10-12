import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    int abs(int i)

@cython.boundscheck(False)
@cython.wraparound(False)
def bresenham(int y, int x, int y2, int x2):
    """
    Generate line pixel coordinates.
    
    Parameters
    ----------
    y, x : int
        Starting position (row, column).
    y2, x2 : int
        End position (row, column).

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the line.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    """
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] rr, cc

    cdef int steep = 0
    cdef int dx = abs(x2 - x)
    cdef int dy = abs(y2 - y)
    cdef int sx, sy, d, i

    if (x2 - x) > 0: sx = 1
    else: sx = -1
    if (y2 - y) > 0: sy = 1
    else: sy = -1
    if dy > dx:
        steep = 1
        x,y = y,x
        dx,dy = dy,dx
        sx,sy = sy,sx
    d = (2 * dy) - dx

    rr = np.zeros(int(dx) + 1, dtype=np.int32)
    cc = np.zeros(int(dx) + 1, dtype=np.int32)
    
    for i in range(dx):
        if steep:
            rr[i] = x
            cc[i] = y
        else:
            rr[i] = y
            cc[i] = x
        while d >= 0:
            y = y + sy
            d = d - (2 * dx)
        x = x + sx
        d = d + (2 * dy)

    rr[dx] = y2
    cc[dx] = x2

    return rr, cc
