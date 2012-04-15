import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    int abs(int i)
    double floor(double x)
    double ceil(double x)
    double round(double x)

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

@cython.boundscheck(False)
@cython.wraparound(False)
def _fill_polygon(
    np.ndarray[np.uint8_t, ndim=2] image,
    np.ndarray[np.double_t, ndim=2] coords,
    int color
):
    cdef int x, y, i, node_idx, swap
    cdef int rows = image.shape[0]
    cdef int cols = image.shape[1]
    cdef int miny = <int>max(0, floor(coords[:,1].min()))
    cdef int maxy = <int>min(rows, ceil(coords[:,1].max()))
    cdef int num_coords = coords.shape[0]
    # array containing the x positions of the intersections of line - polygon
    cdef np.ndarray[np.uint32_t, ndim=1] nodes = np.zeros((num_coords-1,),
        dtype=np.uint32)

    for y in xrange(miny, maxy+1):
        #: determine all intersections of line with polygon
        node_idx = 0
        for i in xrange(0, num_coords-1):
            if (
                (coords[i,1] < y and coords[i+1,1] >= y)
                or (coords[i,1] >= y and coords[i+1,1] < y)
            ):
                nodes[node_idx] = <int>round(
                    (coords[i,0]+(y-coords[i,1])
                    / (coords[i+1,1]-coords[i,1])*(coords[i+1,0]-coords[i,0])))
                node_idx += 1
        # no intersection in current line
        if node_idx == 0:
            continue
        #: bubble sort intersections according to x position on line
        i = 0
        while i < node_idx-1:
            if nodes[i] > nodes[i+1]:
                swap = nodes[i]
                nodes[i] = nodes[i+1]
                nodes[i+1] = swap
                if i:
                    i -= 1
            else:
                i += 1
        #: fill all pixels in current line
        for i in xrange(0, node_idx, 2):
            if i > node_idx and nodes[i] >= cols:
                break
            if nodes[i] < 0:
                nodes[i] = 0
            if nodes[i+1] > cols:
                nodes[i+1] = cols
            for x in xrange(nodes[i], nodes[i+1]):
                image[y,x] = color
