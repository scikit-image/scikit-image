#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import math
import numpy as np

cimport numpy as cnp
from libc.math cimport sqrt
from skimage._shared.geometry cimport point_in_polygon


def line(Py_ssize_t y, Py_ssize_t x, Py_ssize_t y2, Py_ssize_t x2):
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

    cdef cnp.ndarray[cnp.intp_t, ndim=1, mode="c"] rr, cc

    cdef char steep = 0
    cdef Py_ssize_t dx = abs(x2 - x)
    cdef Py_ssize_t dy = abs(y2 - y)
    cdef Py_ssize_t sx, sy, d, i

    if (x2 - x) > 0:
        sx = 1
    else:
        sx = -1
    if (y2 - y) > 0:
        sy = 1
    else:
        sy = -1
    if dy > dx:
        steep = 1
        x, y = y, x
        dx, dy = dy, dx
        sx, sy = sy, sx
    d = (2 * dy) - dx

    rr = np.zeros(int(dx) + 1, dtype=np.intp)
    cc = np.zeros(int(dx) + 1, dtype=np.intp)

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

    cdef Py_ssize_t nr_verts = x.shape[0]
    cdef Py_ssize_t minr = int(max(0, y.min()))
    cdef Py_ssize_t maxr = int(math.ceil(y.max()))
    cdef Py_ssize_t minc = int(max(0, x.min()))
    cdef Py_ssize_t maxc = int(math.ceil(x.max()))

    # make sure output coordinates do not exceed image size
    if shape is not None:
        maxr = min(shape[0] - 1, maxr)
        maxc = min(shape[1] - 1, maxc)

    cdef Py_ssize_t r, c

    # make contigous arrays for r, c coordinates
    cdef cnp.ndarray contiguous_rdata, contiguous_cdata
    contiguous_rdata = np.ascontiguousarray(y, 'double')
    contiguous_cdata = np.ascontiguousarray(x, 'double')
    cdef cnp.double_t* rptr = <cnp.double_t*>contiguous_rdata.data
    cdef cnp.double_t* cptr = <cnp.double_t*>contiguous_cdata.data

    # output coordinate arrays
    cdef list rr = list()
    cdef list cc = list()

    for r in range(minr, maxr+1):
        for c in range(minc, maxc+1):
            if point_in_polygon(nr_verts, cptr, rptr, c, r):
                rr.append(r)
                cc.append(c)

    return np.array(rr, dtype=np.intp), np.array(cc, dtype=np.intp)


def ellipse(double cy, double cx, double yradius, double xradius, shape=None):
    """Generate coordinates of pixels within ellipse.

    Parameters
    ----------
    cy, cx : double
        Centre coordinate of ellipse.
    yradius, xradius : double
        Minor and major semi-axes. ``(x/xradius)**2 + (y/yradius)**2 = 1``.
    shape : tuple, optional
        image shape which is used to determine maximum extents of output pixel
        coordinates. This is useful for ellipses which exceed the image size.
        By default the full extents of the ellipse are used.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of ellipse.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    """

    cdef Py_ssize_t minr = int(max(0, cy - yradius))
    cdef Py_ssize_t maxr = int(math.ceil(cy + yradius))
    cdef Py_ssize_t minc = int(max(0, cx - xradius))
    cdef Py_ssize_t maxc = int(math.ceil(cx + xradius))

    # make sure output coordinates do not exceed image size
    if shape is not None:
        maxr = min(shape[0] - 1, maxr)
        maxc = min(shape[1] - 1, maxc)

    cdef Py_ssize_t r, c

    # output coordinate arrays
    cdef list rr = list()
    cdef list cc = list()

    for r in range(minr, maxr+1):
        for c in range(minc, maxc+1):
            if sqrt(((r - cy) / yradius)**2 + ((c - cx) / xradius)**2) < 1:
                rr.append(r)
                cc.append(c)

    return np.array(rr, dtype=np.intp), np.array(cc, dtype=np.intp)


def circle(double cy, double cx, double radius, shape=None):
    """Generate coordinates of pixels within circle.

    Parameters
    ----------
    cy, cx : double
        Centre coordinate of circle.
    radius: double
        Radius of circle.
    shape : tuple, optional
        image shape which is used to determine maximum extents of output pixel
        coordinates. This is useful for circles which exceed the image size.
        By default the full extents of the circle are used.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of circle.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    Notes
    -----
        This function is a wrapper for skimage.draw.ellipse()
    """

    return ellipse(cy, cx, radius, radius, shape)


def circle_perimeter(Py_ssize_t cy, Py_ssize_t cx, Py_ssize_t radius,
                     method='bresenham'):
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

    cdef Py_ssize_t x = 0
    cdef Py_ssize_t y = radius
    cdef Py_ssize_t d = 0
    cdef char cmethod
    if method == 'bresenham':
        d = 3 - 2 * radius
        cmethod = 'b'
    elif method == 'andres':
        d = radius - 1
        cmethod = 'a'
    else:
        raise ValueError('Wrong method')

    while y >= x:
        rr.extend([y, -y, y, -y, x, -x, x, -x])
        cc.extend([x, x, -x, -x, y, y, -y, -y])

        if cmethod == 'b':
            if d < 0:
                d += 4 * x + 6
            else:
                d += 4 * (x - y) + 10
                y -= 1
            x += 1
        elif cmethod == 'a':
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

    return np.array(rr, dtype=np.intp) + cy, np.array(cc, dtype=np.intp) + cx


def ellipse_perimeter(Py_ssize_t cy, Py_ssize_t cx, Py_ssize_t yradius,
                      Py_ssize_t xradius):
    """Generate ellipse perimeter coordinates.

    Parameters
    ----------
    cy, cx : int
        Centre coordinate of ellipse.
    yradius, xradius: int
        Main radial values.

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the circle perimeter.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    References
    ----------
    .. [1] J. Kennedy "A fast Bresenham type algorithm for
        drawing ellipses".

    """

    # If both radii == 0, return the center to avoid infinite loop in 2nd set
    if xradius == 0 and yradius == 0:
        return np.array(cy), np.array(cx)

    # a and b are xradius an yradius compute 2a^2 and 2b^2
    cdef Py_ssize_t twoasquared = 2 * xradius**2
    cdef Py_ssize_t twobsquared = 2 * yradius**2

    # Pixels
    cdef list px = list()
    cdef list py = list()

    # First set of points:
    # start at the top
    cdef Py_ssize_t x = xradius
    cdef Py_ssize_t y = 0

    cdef Py_ssize_t err = 0
    cdef Py_ssize_t xstop = twobsquared * xradius
    cdef Py_ssize_t ystop = 0
    cdef Py_ssize_t xchange = yradius * yradius * (1 - 2 * xradius)
    cdef Py_ssize_t ychange = xradius * xradius

    while xstop > ystop:
        px.extend([x, -x, -x, x])
        py.extend([y, y, -y, -y])
        y += 1
        ystop += twoasquared
        err += ychange
        ychange += twoasquared
        if (2 * err + xchange) > 0:
            x -= 1
            xstop -= twobsquared
            err += xchange
            xchange += twobsquared

    # Second set of points:
    x = 0
    y = yradius

    err = 0
    xstop = 0
    ystop = twoasquared * yradius
    xchange = yradius * yradius
    ychange = xradius * xradius * (1 - 2 * yradius)

    while xstop <= ystop:
        px.extend([x, -x, -x, x])
        py.extend([y, y, -y, -y])
        x += 1
        xstop += twobsquared
        err += xchange
        xchange += twobsquared
        if  (2 * err + ychange) > 0:
            y -= 1
            ystop -= twoasquared
            err += ychange
            ychange += twobsquared

    return np.array(py, dtype=np.intp) + cy, np.array(px, dtype=np.intp) + cx


def set_color(img, coords, color):
    """Set pixel color in the image at the given coordinates.

    Coordinates that exceed the shape of the image will be ignored.

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
