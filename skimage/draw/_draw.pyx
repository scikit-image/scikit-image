#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import math
import numpy as np

cimport numpy as cnp
from libc.math cimport sqrt, sin, cos, floor, ceil
from .._shared.geometry cimport point_in_polygon


def _coords_inside_image(rr, cc, shape, val=None):
    """
    Return the coordinates inside an image of a given shape.

    Parameters
    ----------
    rr, cc : (N,) ndarray of int
        Indices of pixels.
    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates.
    val : ndarray of float, optional
        Values of pixels at coordinates [rr, cc].

    Returns
    -------
    rr, cc : (N,) array of int
        Row and column indices of valid pixels (i.e. those inside `shape`).
    val : (N,) array of float, optional
        Values at `rr, cc`. Returned only if `val` is given as input.
    """
    mask = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    if val is not None:
        return rr[mask], cc[mask], val[mask]
    return rr[mask], cc[mask]


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

    Notes
    -----
    Anti-aliased line generator is available with `line_aa`.

    Examples
    --------
    >>> from skimage.draw import line
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = line(1, 1, 8, 8)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    """

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

    cdef Py_ssize_t[::1] rr = np.zeros(int(dx) + 1, dtype=np.intp)
    cdef Py_ssize_t[::1] cc = np.zeros(int(dx) + 1, dtype=np.intp)

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

    return np.asarray(rr), np.asarray(cc)


def line_aa(Py_ssize_t y1, Py_ssize_t x1, Py_ssize_t y2, Py_ssize_t x2):
    """Generate anti-aliased line pixel coordinates.

    Parameters
    ----------
    y1, x1 : int
        Starting position (row, column).
    y2, x2 : int
        End position (row, column).

    Returns
    -------
    rr, cc, val : (N,) ndarray (int, int, float)
        Indices of pixels (`rr`, `cc`) and intensity values (`val`).
        ``img[rr, cc] = val``.

    References
    ----------
    .. [1] A Rasterizing Algorithm for Drawing Curves, A. Zingl, 2012
           http://members.chello.at/easyfilter/Bresenham.pdf

    Examples
    --------
    >>> from skimage.draw import line_aa
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc, val = line_aa(1, 1, 8, 8)
    >>> img[rr, cc] = val * 255
    >>> img
    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0, 255,  56,   0,   0,   0,   0,   0,   0,   0],
           [  0,  56, 255,  56,   0,   0,   0,   0,   0,   0],
           [  0,   0,  56, 255,  56,   0,   0,   0,   0,   0],
           [  0,   0,   0,  56, 255,  56,   0,   0,   0,   0],
           [  0,   0,   0,   0,  56, 255,  56,   0,   0,   0],
           [  0,   0,   0,   0,   0,  56, 255,  56,   0,   0],
           [  0,   0,   0,   0,   0,   0,  56, 255,  56,   0],
           [  0,   0,   0,   0,   0,   0,   0,  56, 255,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=uint8)
    """
    cdef list rr = list()
    cdef list cc = list()
    cdef list val = list()

    cdef int dx = abs(x1 - x2)
    cdef int dy = abs(y1 - y2)
    cdef int err = dx - dy
    cdef int x, y, e, ed, sign_x, sign_y

    if x1 < x2:
        sign_x = 1
    else:
        sign_x = -1

    if y1 < y2:
        sign_y = 1
    else:
        sign_y = -1

    if dx + dy == 0:
        ed = 1
    else:
        ed = <int>(sqrt(dx*dx + dy*dy))

    x, y = x1, y1
    while True:
        cc.append(x)
        rr.append(y)
        val.append(1. * abs(err - dx + dy) / <float>(ed))
        e = err
        if 2 * e >= -dx:
            if x == x2:
                break
            if e + dy < ed:
                cc.append(x)
                rr.append(y + sign_y)
                val.append(1. * abs(e + dy) / <float>(ed))
            err -= dy
            x += sign_x
        if 2 * e <= dy:
            if y == y2:
                break
            if dx - e < ed:
                cc.append(x)
                rr.append(y)
                val.append(abs(dx - e) / <float>(ed))
            err += dx
            y += sign_y

    return (np.array(rr, dtype=np.intp),
            np.array(cc, dtype=np.intp),
            1. - np.array(val, dtype=np.float))


def polygon(y, x, shape=None):
    """Generate coordinates of pixels within polygon.

    Parameters
    ----------
    y : (N,) ndarray
        Y-coordinates of vertices of polygon.
    x : (N,) ndarray
        X-coordinates of vertices of polygon.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for polygons which exceed the image
        size. By default the full extent of the polygon are used.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of polygon.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Examples
    --------
    >>> from skimage.draw import polygon
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> x = np.array([1, 7, 4, 1])
    >>> y = np.array([1, 2, 8, 1])
    >>> rr, cc = polygon(y, x)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    """

    cdef Py_ssize_t nr_verts = x.shape[0]
    cdef Py_ssize_t minr = int(max(0, y.min()))
    cdef Py_ssize_t maxr = int(ceil(y.max()))
    cdef Py_ssize_t minc = int(max(0, x.min()))
    cdef Py_ssize_t maxc = int(ceil(x.max()))

    # make sure output coordinates do not exceed image size
    if shape is not None:
        maxr = min(shape[0] - 1, maxr)
        maxc = min(shape[1] - 1, maxc)

    cdef Py_ssize_t r, c

    # make contigous arrays for r, c coordinates
    cdef cnp.ndarray contiguous_rdata, contiguous_cdata
    contiguous_rdata = np.ascontiguousarray(y, dtype=np.double)
    contiguous_cdata = np.ascontiguousarray(x, dtype=np.double)
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


def circle_perimeter(Py_ssize_t cy, Py_ssize_t cx, Py_ssize_t radius,
                     method='bresenham', shape=None):
    """Generate circle perimeter coordinates.

    Parameters
    ----------
    cy, cx : int
        Centre coordinate of circle.
    radius: int
        Radius of circle.
    method : {'bresenham', 'andres'}, optional
        bresenham : Bresenham method (default)
        andres : Andres method
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output pixel
        coordinates. This is useful for circles which exceed the image size.
        By default the full extent of the circle are used.

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Bresenham and Andres' method:
        Indices of pixels that belong to the circle perimeter.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Notes
    -----
    Andres method presents the advantage that concentric
    circles create a disc whereas Bresenham can make holes. There
    is also less distortions when Andres circles are rotated.
    Bresenham method is also known as midpoint circle algorithm.
    Anti-aliased circle generator is available with `circle_perimeter_aa`.

    References
    ----------
    .. [1] J.E. Bresenham, "Algorithm for computer control of a digital
           plotter", IBM Systems journal, 4 (1965) 25-30.
    .. [2] E. Andres, "Discrete circles, rings and spheres", Computers &
           Graphics, 18 (1994) 695-706.

    Examples
    --------
    >>> from skimage.draw import circle_perimeter
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = circle_perimeter(4, 4, 3)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    """

    cdef list rr = list()
    cdef list cc = list()

    cdef Py_ssize_t x = 0
    cdef Py_ssize_t y = radius
    cdef Py_ssize_t d = 0

    cdef double dceil = 0
    cdef double dceil_prev = 0

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
    if shape is not None:
        return _coords_inside_image(np.array(rr, dtype=np.intp) + cy,
                                    np.array(cc, dtype=np.intp) + cx,
                                    shape)
    return (np.array(rr, dtype=np.intp) + cy,
            np.array(cc, dtype=np.intp) + cx)


def circle_perimeter_aa(Py_ssize_t cy, Py_ssize_t cx, Py_ssize_t radius,
                        shape=None):
    """Generate anti-aliased circle perimeter coordinates.

    Parameters
    ----------
    cy, cx : int
        Centre coordinate of circle.
    radius: int
        Radius of circle.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output pixel
        coordinates. This is useful for circles which exceed the image size.
        By default the full extent of the circle are used.

    Returns
    -------
    rr, cc, val : (N,) ndarray (int, int, float)
        Indices of pixels (`rr`, `cc`) and intensity values (`val`).
        ``img[rr, cc] = val``.

    Notes
    -----
    Wu's method draws anti-aliased circle. This implementation doesn't use
    lookup table optimization.

    References
    ----------
    .. [1] X. Wu, "An efficient antialiasing technique", In ACM SIGGRAPH
           Computer Graphics, 25 (1991) 143-152.

    Examples
    --------
    >>> from skimage.draw import circle_perimeter_aa
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc, val = circle_perimeter_aa(4, 4, 3)
    >>> img[rr, cc] = val * 255
    >>> img
    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,  60, 211, 255, 211,  60,   0,   0,   0],
           [  0,  60, 194,  43,   0,  43, 194,  60,   0,   0],
           [  0, 211,  43,   0,   0,   0,  43, 211,   0,   0],
           [  0, 255,   0,   0,   0,   0,   0, 255,   0,   0],
           [  0, 211,  43,   0,   0,   0,  43, 211,   0,   0],
           [  0,  60, 194,  43,   0,  43, 194,  60,   0,   0],
           [  0,   0,  60, 211, 255, 211,  60,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=uint8)
    """

    cdef Py_ssize_t x = 0
    cdef Py_ssize_t y = radius
    cdef Py_ssize_t d = 0

    cdef double dceil = 0
    cdef double dceil_prev = 0

    cdef list rr = [y, x,  y,  x, -y, -x, -y, -x]
    cdef list cc = [x, y, -x, -y,  x,  y, -x, -y]
    cdef list val = [1] * 8

    while y > x + 1:
        x += 1
        dceil = sqrt(radius**2 - x**2)
        dceil = ceil(dceil) - dceil
        if dceil < dceil_prev:
            y -= 1
        rr.extend([y, y - 1, x, x, y, y - 1, x, x])
        cc.extend([x, x, y, y - 1, -x, -x, -y, 1 - y])

        rr.extend([-y, 1 - y, -x, -x, -y, 1 - y, -x, -x])
        cc.extend([x, x, y, y - 1, -x, -x, -y, 1 - y])

        val.extend([1 - dceil, dceil] * 8)
        dceil_prev = dceil

    if shape is not None:
        return _coords_inside_image(np.array(rr, dtype=np.intp) + cy,
                                    np.array(cc, dtype=np.intp) + cx,
                                    shape,
                                    val=np.array(val, dtype=np.float))
    return (np.array(rr, dtype=np.intp) + cy,
            np.array(cc, dtype=np.intp) + cx,
            np.array(val, dtype=np.float))


def ellipse_perimeter(Py_ssize_t cy, Py_ssize_t cx, Py_ssize_t yradius,
                      Py_ssize_t xradius, double orientation=0, shape=None):
    """Generate ellipse perimeter coordinates.

    Parameters
    ----------
    cy, cx : int
        Centre coordinate of ellipse.
    yradius, xradius : int
        Minor and major semi-axes. ``(x/xradius)**2 + (y/yradius)**2 = 1``.
    orientation : double, optional (default 0)
        Major axis orientation in clockwise direction as radians.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output pixel
        coordinates. This is useful for ellipses which exceed the image size.
        By default the full extent of the ellipse are used.

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the ellipse perimeter.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    References
    ----------
    .. [1] A Rasterizing Algorithm for Drawing Curves, A. Zingl, 2012
           http://members.chello.at/easyfilter/Bresenham.pdf

    Examples
    --------
    >>> from skimage.draw import ellipse_perimeter
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = ellipse_perimeter(5, 5, 3, 4)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    """

    # If both radii == 0, return the center to avoid infinite loop in 2nd set
    if xradius == 0 and yradius == 0:
        return np.array(cy), np.array(cx)

    # Pixels
    cdef list px = list()
    cdef list py = list()

    # Compute useful values
    cdef  Py_ssize_t xd = xradius**2
    cdef  Py_ssize_t yd = yradius**2

    cdef Py_ssize_t x, y, e2, err

    cdef int ix0, ix1, iy0, iy1, ixd, iyd
    cdef double sin_angle, xa, ya, za, a, b

    if orientation == 0:
        x = -xradius
        y = 0
        e2 = yd
        err = x*(2 * e2 + x) + e2
        while x <= 0:
            # Quadrant 1
            px.append(cx - x)
            py.append(cy + y)
            # Quadrant 2
            px.append(cx + x)
            py.append(cy + y)
            # Quadrant 3
            px.append(cx + x)
            py.append(cy - y)
            # Quadrant 4
            px.append(cx - x)
            py.append(cy - y)
            # Adjust x and y
            e2 = 2 * err
            if e2 >= (2 * x + 1) * yd:
                x += 1
                err += (2 * x + 1) * yd
            if e2 <= (2 * y + 1) * xd:
                y += 1
                err += (2 * y + 1) * xd
        while y < yradius:
            y += 1
            px.append(cx)
            py.append(cy + y)
            px.append(cx)
            py.append(cy - y)

    else:
        sin_angle = sin(orientation)
        za = (xd - yd) * sin_angle
        xa = sqrt(xd - za * sin_angle)
        ya = sqrt(yd + za * sin_angle)

        a = xa + 0.5
        b = ya + 0.5
        za = za * a * b / (xa * ya)

        ix0 = int(cx - a)
        iy0 = int(cy - b)
        ix1 = int(cx + a)
        iy1 = int(cy + b)

        xa = ix1 - ix0
        ya = iy1 - iy0
        za = 4 * za * cos(orientation)
        w = xa * ya
        if w != 0:
            w = (w - za) / (w + w)
        ixd = int(floor(xa * w + 0.5))
        iyd = int(floor(ya * w + 0.5))

        # Draw the 4 quadrants
        rr, cc = _bezier_segment(iy0 + iyd, ix0, iy0, ix0, iy0, ix0 + ixd, 1-w)
        py.extend(rr)
        px.extend(cc)
        rr, cc = _bezier_segment(iy0 + iyd, ix0, iy1, ix0, iy1, ix1 - ixd, w)
        py.extend(rr)
        px.extend(cc)
        rr, cc = _bezier_segment(iy1 - iyd, ix1, iy1, ix1, iy1, ix1 - ixd, 1-w)
        py.extend(rr)
        px.extend(cc)
        rr, cc = _bezier_segment(iy1 - iyd, ix1, iy0, ix1, iy0, ix0 + ixd,  w)
        py.extend(rr)
        px.extend(cc)

    if shape is not None:
        return _coords_inside_image(np.array(py, dtype=np.intp),
                                    np.array(px, dtype=np.intp), shape)
    return np.array(py, dtype=np.intp), np.array(px, dtype=np.intp)


def _bezier_segment(Py_ssize_t y0, Py_ssize_t x0,
                    Py_ssize_t y1, Py_ssize_t x1,
                    Py_ssize_t y2, Py_ssize_t x2,
                    double weight):
    """Generate Bezier segment coordinates.

    Parameters
    ----------
    y0, x0 : int
        Coordinates of the first control point.
    y1, x1 : int
        Coordinates of the middle control point.
    y2, x2 : int
        Coordinates of the last control point.
    weight : double
        Middle control point weight, it describes the line tension.

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the Bezier curve.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Notes
    -----
    The algorithm is the rational quadratic algorithm presented in
    reference [1]_.

    References
    ----------
    .. [1] A Rasterizing Algorithm for Drawing Curves, A. Zingl, 2012
           http://members.chello.at/easyfilter/Bresenham.pdf
    """
    # Pixels
    cdef list px = list()
    cdef list py = list()

    # Steps
    cdef double sx = x2 - x1
    cdef double sy = y2 - y1

    cdef double dx = x0 - x2
    cdef double dy = y0 - y2
    cdef double xx = x0 - x1
    cdef double yy = y0 - y1
    cdef double xy = xx * sy + yy * sx
    cdef double cur = xx * sy - yy * sx
    cdef double err

    cdef bint test1, test2

    # if it's not a straight line
    if cur != 0 and weight > 0:
        if (sx * sx + sy * sy > xx * xx + yy * yy):
            # Swap point 0 and point 2
            # to start from the longer part
            x2 = x0
            x0 -= <Py_ssize_t>(dx)
            y2 = y0
            y0 -= <Py_ssize_t>(dy)
            cur = -cur
        xx = 2 * (4 * weight * sx * xx + dx * dx)
        yy = 2 * (4 * weight * sy * yy + dy * dy)
        # Set steps
        if x0 < x2:
            sx = 1
        else:
            sx = -1
        if y0 < y2:
            sy = 1
        else:
            sy = -1
        xy = -2 * sx * sy * (2 * weight * xy + dx * dy)

        if cur * sx * sy < 0:
            xx = -xx
            yy = -yy
            xy = -xy
            cur = -cur

        dx = 4 * weight * (x1 - x0) * sy * cur + xx / 2 + xy
        dy = 4 * weight * (y0 - y1) * sx * cur + yy / 2 + xy

        # Flat ellipse, algo fails
        if (weight < 0.5 and (dy > xy or dx < xy)):
            cur = (weight + 1) / 2
            weight = sqrt(weight)
            xy = 1. / (weight + 1)
            # subdivide curve in half
            sx = floor((x0 + 2 * weight * x1 + x2) * xy * 0.5 + 0.5)
            sy = floor((y0 + 2 * weight * y1 + y2) * xy * 0.5 + 0.5)
            dx = floor((weight * x1 + x0) * xy + 0.5)
            dy = floor((y1 * weight + y0) * xy + 0.5)
            return _bezier_segment(y0, x0, <Py_ssize_t>(dy), <Py_ssize_t>(dx),
                                   <Py_ssize_t>(sy), <Py_ssize_t>(sx), cur)

        err = dx + dy - xy
        while dy <= xy and dx >= xy:
            px.append(x0)
            py.append(y0)
            if x0 == x2 and y0 == y2:
                # The job is done!
                return np.array(py, dtype=np.intp), np.array(px, dtype=np.intp)

            # Save boolean values
            test1 = 2 * err > dy
            test2 = 2 * (err + yy) < -dy
            # Move (x0,y0) to the next position
            if 2 * err < dx or test2:
                y0 += <Py_ssize_t>(sy)
                dy += xy
                dx += xx
                err += dx
            if 2 * err > dx or test1:
                x0 += <Py_ssize_t>(sx)
                dx += xy
                dy += yy
                err += dy

    # Plot line
    rr, cc = line(x0, y0, x2, y2)
    px.extend(rr)
    py.extend(cc)

    return np.array(py, dtype=np.intp), np.array(px, dtype=np.intp)


def bezier_curve(Py_ssize_t y0, Py_ssize_t x0,
                 Py_ssize_t y1, Py_ssize_t x1,
                 Py_ssize_t y2, Py_ssize_t x2,
                 double weight, shape=None):
    """Generate Bezier curve coordinates.

    Parameters
    ----------
    y0, x0 : int
        Coordinates of the first control point.
    y1, x1 : int
        Coordinates of the middle control point.
    y2, x2 : int
        Coordinates of the last control point.
    weight : double
        Middle control point weight, it describes the line tension.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for curves which exceed the image
        size. By default the full extent of the curve are used.

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the Bezier curve.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Notes
    -----
    The algorithm is the rational quadratic algorithm presented in
    reference [1]_.

    References
    ----------
    .. [1] A Rasterizing Algorithm for Drawing Curves, A. Zingl, 2012
           http://members.chello.at/easyfilter/Bresenham.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.draw import bezier_curve
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = bezier_curve(1, 5, 5, -2, 8, 8, 2)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    # Pixels
    cdef list px = list()
    cdef list py = list()

    cdef int x, y
    cdef double xx, yy, ww, t, q
    x = x0 - 2 * x1 + x2
    y = y0 - 2 * y1 + y2

    xx = x0 - x1
    yy = y0 - y1

    if xx * (x2 - x1) > 0:
        if yy * (y2 - y1):
            if abs(xx * y) > abs(yy * x):
                x0 = x2
                x2 = <Py_ssize_t>(xx + x1)
                y0 = y2
                y2 = <Py_ssize_t>(yy + y1)
        if (x0 == x2) or (weight == 1.):
            t = <double>(x0 - x1) / x
        else:
            q = sqrt(4. * weight * weight * (x0 - x1) * (x2 - x1) + (x2 - x0) * floor(x2 - x0))
            if (x1 < x0):
                q = -q
            t = (2. * weight * (x0 - x1) - x0 + x2 + q) / (2. * (1. - weight) * (x2 - x0))

        q = 1. / (2. * t * (1. - t) * (weight - 1.) + 1.0)
        xx = (t * t * (x0 - 2. * weight * x1 + x2) + 2. * t * (weight * x1 - x0) + x0) * q
        yy = (t * t * (y0 - 2. * weight * y1 + y2) + 2. * t * (weight * y1 - y0) + y0) * q
        ww = t * (weight - 1.) + 1.
        ww *= ww * q
        weight = ((1. - t) * (weight - 1.) + 1.) * sqrt(q)
        x = <int>(xx + 0.5)
        y = <int>(yy + 0.5)
        yy = (xx - x0) * (y1 - y0) / (x1 - x0) + y0

        rr, cc = _bezier_segment(y0, x0, <int>(yy + 0.5), x, y, x, ww)
        px.extend(rr)
        py.extend(cc)

        yy = (xx - x2) * (y1 - y2) / (x1 - x2) + y2
        y1 = <int>(yy + 0.5)
        x0 = x1 = x
        y0 = y
    if (y0 - y1) * floor(y2 - y1) > 0:
        if (y0 == y2) or (weight == 1):
            t = (y0 - y1) / (y0 - 2. * y1 + y2)
        else:
            q = sqrt(4. * weight * weight * (y0 - y1) * (y2 - y1) + (y2 - y0) * floor(y2 - y0))
            if y1 < y0:
                q = -q
            t = (2. * weight * (y0 - y1) - y0 + y2 + q) / (2. * (1. - weight) * (y2 - y0))
        q = 1. / (2. * t * (1. - t) * (weight - 1.) + 1.)
        xx = (t * t * (x0 - 2. * weight * x1 + x2) + 2. * t * (weight * x1 - x0) + x0) * q
        yy = (t * t * (y0 - 2. * weight * y1 + y2) + 2. * t * (weight * y1 - y0) + y0) * q
        ww = t * (weight - 1.) + 1.
        ww *= ww * q
        weight = ((1. - t) * (weight - 1.) + 1.) * sqrt(q)
        x = <int>(xx + 0.5)
        y = <int>(yy + 0.5)
        xx = (x1 - x0) * (yy - y0) / (y1 - y0) + x0

        rr, cc = _bezier_segment(y0, x0, y, <int>(xx + 0.5), y, x, ww)
        px.extend(rr)
        py.extend(cc)

        xx = (x1 - x2) * (yy - y2) / (y1 - y2) + x2
        x1 = <int>(xx + 0.5)
        x0 = x
        y0 = y1 = y

    rr, cc = _bezier_segment(y0, x0, y1, x1, y2, x2, weight * weight)
    px.extend(rr)
    py.extend(cc)

    if shape is not None:
        return _coords_inside_image(np.array(px, dtype=np.intp),
                                    np.array(py, dtype=np.intp), shape)
    return np.array(px, dtype=np.intp), np.array(py, dtype=np.intp)
