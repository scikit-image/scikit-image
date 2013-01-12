import numpy as np
import math
from libc.math cimport sqrt
cimport numpy as np
cimport cython
from skimage._shared.geometry cimport point_in_polygon


@cython.boundscheck(False)
@cython.wraparound(False)
def line(int y, int x, int y2, int x2):
    """Generate line pixel coordinates.

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
@cython.nonecheck(False)
def polygon(y, x, shape=None):
    """Generate coordinates of pixels within polygon.

    Parameters
    ----------
    y : (N,) ndarray
        Y-coordinates of vertices of polygon.
    x : (N,) ndarray
        X-coordinates of vertices of polygon.
    shape : tuple, optional
        image shape which is used to determine maximum extents of output pixel
        coordinates. This is useful for polygons which exceed the image size.
        By default the full extents of the polygon are used.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of polygon.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    """
    cdef int nr_verts = x.shape[0]
    cdef int minr = <int>max(0, y.min())
    cdef int maxr = <int>math.ceil(y.max())
    cdef int minc = <int>max(0, x.min())
    cdef int maxc = <int>math.ceil(x.max())

    # make sure output coordinates do not exceed image size
    if shape is not None:
        maxr = min(shape[0]-1, maxr)
        maxc = min(shape[1]-1, maxc)

    cdef int r, c

    #: make contigous arrays for r, c coordinates
    cdef np.ndarray contiguous_rdata, contiguous_cdata
    contiguous_rdata = np.ascontiguousarray(y, 'double')
    contiguous_cdata = np.ascontiguousarray(x, 'double')
    cdef np.double_t* rptr = <np.double_t*>contiguous_rdata.data
    cdef np.double_t* cptr = <np.double_t*>contiguous_cdata.data

    #: output coordinate arrays
    cdef list rr = list()
    cdef list cc = list()

    for r in range(minr, maxr+1):
        for c in range(minc, maxc+1):
            if point_in_polygon(nr_verts, cptr, rptr, c, r):
                rr.append(r)
                cc.append(c)

    return np.array(rr), np.array(cc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def ellipse(double cy, double cx, double b, double a, shape=None):
    """Generate coordinates of pixels within ellipse.

    Parameters
    ----------
    cy, cx : double
        Centre coordinate of ellipse.
    b, a: double
        Minor and major semi-axes. ``(x/a)**2 + (y/b)**2 = 1``.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of ellipse.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    """
    cdef int minr = <int>max(0, cy-b)
    cdef int maxr = <int>math.ceil(cy+b)
    cdef int minc = <int>max(0, cx-a)
    cdef int maxc = <int>math.ceil(cx+a)

    # make sure output coordinates do not exceed image size
    if shape is not None:
        maxr = min(shape[0]-1, maxr)
        maxc = min(shape[1]-1, maxc)

    cdef int r, c

    #: output coordinate arrays
    cdef list rr = list()
    cdef list cc = list()

    for r in range(minr, maxr+1):
        for c in range(minc, maxc+1):
            if sqrt(((r - cy)/b)**2 + ((c - cx)/a)**2) < 1:
                rr.append(r)
                cc.append(c)

    return np.array(rr), np.array(cc)


def circle(double cy, double cx, double radius, shape=None):
    """Generate coordinates of pixels within circle.

    Parameters
    ----------
    cy, cx : double
        Centre coordinate of circle.
    radius: double
        Radius of circle.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of circle.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    """
    return ellipse(cy, cx, radius, radius, shape)


def circle_perimeter(int cy, int cx, int radius, method='bresenham'):
    """Generate circle perimeter coordinates.

    Parameters
    ----------
    cy, cx : int
        Centre coordinate of circle.
    radius: int
        Radius of circle.
    method : {'bresenham', 'andres'}, optional
        bresenham : Bresenham method
        andres : Andres method


    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the circle perimeter.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Notes
    -----
    Andres method presents the advantage that concentric
    circles create a disc whereas Bresenham can make holes. There
    is also less distortions when Andres circles are rotated.
    Bresenham method is also known as midpoint circle algorithm.

    References
    ----------
    .. [1] J.E. Bresenham, "Algorithm for computer control of a digital
        plotter", 4 (1965) 25-30.
    .. [2] E. Andres, "Discrete circles, rings and spheres", 18 (1994) 695-706.

    """

    cdef list rr = list()
    cdef list cc = list()

    cdef int x = 0
    cdef int y = radius
    cdef int d = 0
    if method == 'bresenham':
        d = 3 - 2 * radius
    elif method == 'andres':
        d = radius - 1
    else:
        raise ValueError('Wrong method')

    while y >= x:
        rr.extend([y, -y, y, -y, x, -x, x, -x])
        cc.extend([x, x, -x, -x, y, y, -y, -y])

        if method == 'bresenham':
            if d < 0:
                d += 4 * x + 6
            else:
                d += 4 * (x - y) + 10
                y -= 1
            x += 1
        elif method == 'andres':
            if d >= 2 * (x - 1):
                d = d - 2 * x
                x = x + 1
            elif d <= 2 * (radius - y):
                d = d + 2 * y - 1
                y = y - 1
            else:
                d = d + 2 * (y - x - 1)
                y = y - 1
                x = x + 1

    return np.array(rr) + cy, np.array(cc) + cx


@cython.boundscheck(False)
@cython.wraparound(False)
def set_color(img, coords, color):
    """Set pixel color in the image at the given coordiantes. Coordinates that
    exceeed the shape of the image will be ignored.

    Parameters
    ----------
    img : (M, N, D) ndarray
        Image
    coords : ((P,) ndarray, (P,) ndarray)
        Coordinates of pixels to be colored.
    color : (D,) ndarray
        Color to be assigned to coordinates in the image.

    Returns
    -------
    img : (M, N, D) ndarray
        The updated image.
    """
    rr, cc = coords
    rr_inside = np.logical_and(rr >= 0, rr < img.shape[0])
    cc_inside = np.logical_and(cc >= 0, cc < img.shape[1])
    inside = np.logical_and(rr_inside, cc_inside)
    img[rr[inside], cc[inside]] = color
