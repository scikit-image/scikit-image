#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import math
import numpy as np

cimport numpy as cnp
from libc.math cimport sqrt, sin, cos, floor
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

    cdef Py_ssize_t[:] rr = np.zeros(int(dx) + 1, dtype=np.intp)
    cdef Py_ssize_t[:] cc = np.zeros(int(dx) + 1, dtype=np.intp)

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


def polygon(y, x, shape=None):
    """Generate coordinates of pixels within polygon.

    Parameters
    ----------
    y : (N,) ndarray
        Y-coordinates of vertices of polygon.
    x : (N,) ndarray
        X-coordinates of vertices of polygon.
    shape : tuple, optional
        Image shape which is used to determine maximum extents of output pixel
        coordinates. This is useful for polygons which exceed the image size.
        By default the full extents of the polygon are used.

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
                     method='bresenham'):
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
    .. [2] E. Andres, "Discrete circles, rings and spheres",
           18 (1994) 695-706.

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
                      Py_ssize_t xradius, double orientation=0):
    """Generate ellipse perimeter coordinates.

    Parameters
    ----------
    cy, cx : int
        Centre coordinate of ellipse.
    yradius, xradius: int
        Minor and major semi-axes. ``(x/xradius)**2 + (y/yradius)**2 = 1``.
    orientation: double, optional (default 0)
        Major axis orientation in clockwise direction as radians.

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
        rr, cc = bezier_segment(iy0 + iyd, ix0, iy0, ix0, iy0, ix0 + ixd, 1-w)
        py.extend(rr)
        px.extend(cc)
        rr, cc = bezier_segment(iy0 + iyd, ix0, iy1, ix0, iy1, ix1 - ixd, w)
        py.extend(rr)
        px.extend(cc)
        rr, cc = bezier_segment(iy1 - iyd, ix1, iy1, ix1, iy1, ix1 - ixd, 1-w)
        py.extend(rr)
        px.extend(cc)
        rr, cc = bezier_segment(iy1 - iyd, ix1, iy0, ix1, iy0, ix0 + ixd,  w)
        py.extend(rr)
        px.extend(cc)

    return np.array(py, dtype=np.intp), np.array(px, dtype=np.intp)


def bezier_segment(Py_ssize_t y0, Py_ssize_t x0,
                   Py_ssize_t y1, Py_ssize_t x1,
                   Py_ssize_t y2, Py_ssize_t x2,
                   double weight):
    """Generate Bezier segment coordinates.

    Parameters
    ----------
    y0, x0 : int
        Coordinates of the first point
    y1, x1 : int
        Coordinates of the middle point
    y2, x2 : int
        Coordinates of the last point
    weight : double
        Middle point weight, it describes the line tension.

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the Bezier curve.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Notes
    -----
    The algorithm is the rational quadratic algorithm presented in
    reference [1].

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
            return bezier_segment(y0, x0, <Py_ssize_t>(dy), <Py_ssize_t>(dx),
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
