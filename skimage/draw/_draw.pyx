#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import math
import numpy as np

cimport numpy as cnp
from libc.math cimport sqrt, sin, cos, floor, ceil, fabs
from .._shared.geometry cimport point_in_polygon

cnp.import_array()


def _coords_inside_image(rr, cc, shape, val=None):
    """
    Return the coordinates inside an image of a given shape.

    Parameters
    ----------
    rr, cc : (N,) ndarray of int
        Indices of pixels.
    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates.  Must be at least length 2. Only the first two values
        are used to determine the extent of the input image.
    val : (N, D) ndarray of float, optional
        Values of pixels at coordinates ``[rr, cc]``.

    Returns
    -------
    rr, cc : (M,) array of int
        Row and column indices of valid pixels (i.e. those inside `shape`).
    val : (M, D) array of float, optional
        Values at `rr, cc`. Returned only if `val` is given as input.
    """
    mask = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    if val is None:
        return rr[mask], cc[mask]
    else:
        return rr[mask], cc[mask], val[mask]


def _line(Py_ssize_t r0, Py_ssize_t c0, Py_ssize_t r1, Py_ssize_t c1):
    """Generate line pixel coordinates.

    Parameters
    ----------
    r0, c0 : int
        Starting position (row, column).
    r1, c1 : int
        End position (row, column).

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the line.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    See Also
    --------
    line_aa : Anti-aliased line generator
    """

    cdef char steep = 0
    cdef Py_ssize_t r = r0
    cdef Py_ssize_t c = c0
    cdef Py_ssize_t dr = abs(r1 - r0)
    cdef Py_ssize_t dc = abs(c1 - c0)
    cdef Py_ssize_t sr, sc, d, i

    cdef Py_ssize_t[::1] rr = np.zeros(max(dc, dr) + 1, dtype=np.intp)
    cdef Py_ssize_t[::1] cc = np.zeros(max(dc, dr) + 1, dtype=np.intp)

    with nogil:
        if (c1 - c) > 0:
            sc = 1
        else:
            sc = -1
        if (r1 - r) > 0:
            sr = 1
        else:
            sr = -1
        if dr > dc:
            steep = 1
            c, r = r, c
            dc, dr = dr, dc
            sc, sr = sr, sc
        d = (2 * dr) - dc

        for i in range(dc):
            if steep:
                rr[i] = c
                cc[i] = r
            else:
                rr[i] = r
                cc[i] = c
            while d >= 0:
                r = r + sr
                d = d - (2 * dc)
            c = c + sc
            d = d + (2 * dr)

        rr[dc] = r1
        cc[dc] = c1

    return np.asarray(rr), np.asarray(cc)


def _line_aa(Py_ssize_t r0, Py_ssize_t c0, Py_ssize_t r1, Py_ssize_t c1):
    """Generate anti-aliased line pixel coordinates.

    Parameters
    ----------
    r0, c0 : int
        Starting position (row, column).
    r1, c1 : int
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
    """
    cdef list rr = list()
    cdef list cc = list()
    cdef list val = list()

    cdef int dc = abs(c0 - c1)
    cdef int dc_prime

    cdef int dr = abs(r0 - r1)
    cdef float err = dc - dr
    cdef float err_prime

    cdef int c, r, sign_c, sign_r
    cdef float ed

    if c0 < c1:
        sign_c = 1
    else:
        sign_c = -1

    if r0 < r1:
        sign_r = 1
    else:
        sign_r = -1

    if dc + dr == 0:
        ed = 1
    else:
        ed = sqrt(dc*dc + dr*dr)

    c, r = c0, r0
    while True:
        cc.append(c)
        rr.append(r)
        val.append(fabs(err - dc + dr) / ed)

        err_prime = err
        c_prime = c

        if (2 * err_prime) >= -dc:
            if c == c1:
                break
            if (err_prime + dr) < ed:
                cc.append(c)
                rr.append(r + sign_r)
                val.append(fabs(err_prime + dr) / ed)
            err -= dr
            c += sign_c

        if 2 * err_prime <= dr:
            if r == r1:
                break
            if (dc - err_prime) < ed:
                cc.append(c_prime + sign_c)
                rr.append(r)
                val.append(fabs(dc - err_prime) / ed)
            err += dc
            r += sign_r

    return (np.array(rr, dtype=np.intp),
            np.array(cc, dtype=np.intp),
            1. - np.array(val, dtype=float))


def _polygon(r, c, shape):
    """Generate coordinates of pixels within polygon.

    Parameters
    ----------
    r : (N,) ndarray
        Row coordinates of vertices of polygon.
    c : (N,) ndarray
        Column coordinates of vertices of polygon.
    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for polygons that exceed the image
        size. If None, the full extent of the polygon is used.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of polygon.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    """
    r = np.atleast_1d(r)
    c = np.atleast_1d(c)

    cdef Py_ssize_t nr_verts = c.shape[0]
    cdef Py_ssize_t minr = int(max(0, r.min()))
    cdef Py_ssize_t maxr = int(ceil(r.max()))
    cdef Py_ssize_t minc = int(max(0, c.min()))
    cdef Py_ssize_t maxc = int(ceil(c.max()))

    # make sure output coordinates do not exceed image size
    if shape is not None:
        maxr = min(shape[0] - 1, maxr)
        maxc = min(shape[1] - 1, maxc)

    # make contiguous arrays for r, c coordinates
    cdef cnp.float64_t[::1] rptr = np.ascontiguousarray(r, 'float64')
    cdef cnp.float64_t[::1] cptr = np.ascontiguousarray(c, 'float64')
    cdef cnp.float64_t r_i, c_i

    # output coordinate arrays
    rr = list()
    cc = list()

    for r_i in range(minr, maxr+1):
        for c_i in range(minc, maxc+1):
            if point_in_polygon(cptr, rptr, c_i, r_i):
                rr.append(r_i)
                cc.append(c_i)

    return np.array(rr, dtype=np.intp), np.array(cc, dtype=np.intp)


def _circle_perimeter(Py_ssize_t r_o, Py_ssize_t c_o, Py_ssize_t radius,
                      method, shape):
    """Generate circle perimeter coordinates.

    Parameters
    ----------
    r_o, c_o : int
        Centre coordinate of circle.
    radius : int
        Radius of circle.
    method : {'bresenham', 'andres'}
        bresenham : Bresenham method (default)
        andres : Andres method
    shape : tuple
        Image shape which is used to determine the maximum extent of output pixel
        coordinates. This is useful for circles that exceed the image size.
        If None, the full extent of the circle is used.

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
    """

    cdef list rr = list()
    cdef list cc = list()

    cdef Py_ssize_t c = 0
    cdef Py_ssize_t r = radius
    cdef Py_ssize_t d = 0

    cdef double dceil = 0
    cdef double dceil_prev = 0

    cdef char cmethod
    if method == 'bresenham':
        d = 3 - 2 * radius
        cmethod = b'b'
    elif method == 'andres':
        d = radius - 1
        cmethod = b'a'
    else:
        raise ValueError('Wrong method')

    while r >= c:
        rr.extend([r, -r, r, -r, c, -c, c, -c])
        cc.extend([c, c, -c, -c, r, r, -r, -r])

        if cmethod == b'b':
            if d < 0:
                d += 4 * c + 6
            else:
                d += 4 * (c - r) + 10
                r -= 1
            c += 1
        elif cmethod == b'a':
            if d >= 2 * (c - 1):
                d = d - 2 * c
                c = c + 1
            elif d <= 2 * (radius - r):
                d = d + 2 * r - 1
                r = r - 1
            else:
                d = d + 2 * (r - c - 1)
                r = r - 1
                c = c + 1

    if shape is not None:
        return _coords_inside_image(np.array(rr, dtype=np.intp) + r_o,
                                    np.array(cc, dtype=np.intp) + c_o,
                                    shape)
    return (np.array(rr, dtype=np.intp) + r_o,
            np.array(cc, dtype=np.intp) + c_o)


def _circle_perimeter_aa(Py_ssize_t r_o, Py_ssize_t c_o,
                         Py_ssize_t radius, shape):
    """Generate anti-aliased circle perimeter coordinates.

    Parameters
    ----------
    r_o, c_o : int
        Centre coordinate of circle.
    radius : int
        Radius of circle.
    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for circles that exceed the image
        size. If None, the full extent of the circle is used.

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
    """

    cdef Py_ssize_t c = 0
    cdef Py_ssize_t r = radius
    cdef Py_ssize_t d = 0

    cdef double dceil = 0
    cdef double dceil_prev = 0

    cdef list rr = [r, c,  r,  c, -r, -c, -r, -c]
    cdef list cc = [c, r, -c, -r,  c,  r, -c, -r]
    cdef list val = [1] * 8

    while r > c + 1:
        c += 1
        dceil = sqrt(radius * radius - c * c)
        dceil = ceil(dceil) - dceil
        if dceil < dceil_prev:
            r -= 1
        rr.extend([r, r - 1, c, c, r, r - 1, c, c])
        cc.extend([c, c, r, r - 1, -c, -c, -r, 1 - r])

        rr.extend([-r, 1 - r, -c, -c, -r, 1 - r, -c, -c])
        cc.extend([c, c, r, r - 1, -c, -c, -r, 1 - r])

        val.extend([1 - dceil, dceil] * 8)
        dceil_prev = dceil

    if shape is not None:
        return _coords_inside_image(np.array(rr, dtype=np.intp) + r_o,
                                    np.array(cc, dtype=np.intp) + c_o,
                                    shape,
                                    val=np.array(val, dtype=float))
    return (np.array(rr, dtype=np.intp) + r_o,
            np.array(cc, dtype=np.intp) + c_o,
            np.array(val, dtype=float))


def _ellipse_perimeter(Py_ssize_t r_o, Py_ssize_t c_o, Py_ssize_t r_radius,
                       Py_ssize_t c_radius, double orientation, shape):
    """Generate ellipse perimeter coordinates.

    Parameters
    ----------
    r_o, c_o : int
        Centre coordinate of ellipse.
    r_radius, c_radius : int
        Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 = 1``.
    orientation : double
        Major axis orientation in clockwise direction as radians.
    shape : tuple
        Image shape which is used to determine the maximum extent of output pixel
        coordinates. This is useful for ellipses that exceed the image size.
        If None, the full extent of the ellipse is used.

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
    """

    # If both radii == 0, return the center to avoid infinite loop in 2nd set
    if r_radius == 0 and c_radius == 0:
        return np.array(r_o), np.array(c_o)

    # Pixels
    cdef list rr = list()
    cdef list cc = list()

    # Compute useful values
    cdef  Py_ssize_t rd = r_radius * r_radius
    cdef  Py_ssize_t cd = c_radius * c_radius

    cdef Py_ssize_t r, c, e2, err

    cdef int ir0, ir1, ic0, ic1, ird, icd
    cdef double sin_angle, ra, ca, za, a, b

    if orientation == 0:
        c = -c_radius
        r = 0
        e2 = rd
        err = c * (2 * e2 + c) + e2
        while c <= 0:
            # Quadrant 1
            rr.append(r_o + r)
            cc.append(c_o - c)
            # Quadrant 2
            rr.append(r_o + r)
            cc.append(c_o + c)
            # Quadrant 3
            rr.append(r_o - r)
            cc.append(c_o + c)
            # Quadrant 4
            rr.append(r_o - r)
            cc.append(c_o - c)
            # Adjust `r` and `c`
            e2 = 2 * err
            if e2 >= (2 * c + 1) * rd:
                c += 1
                err += (2 * c + 1) * rd
            if e2 <= (2 * r + 1) * cd:
                r += 1
                err += (2 * r + 1) * cd
        while r < r_radius:
            r += 1
            rr.append(r_o + r)
            cc.append(c_o)
            rr.append(r_o - r)
            cc.append(c_o)

    else:
        sin_angle = sin(orientation)
        za = (cd - rd) * sin_angle
        ca = sqrt(cd - za * sin_angle)
        ra = sqrt(rd + za * sin_angle)

        a = ca + 0.5
        b = ra + 0.5
        za = za * a * b / (ca * ra)

        ir0 = int(r_o - b)
        ic0 = int(c_o - a)
        ir1 = int(r_o + b)
        ic1 = int(c_o + a)

        ca = ic1 - ic0
        ra = ir1 - ir0
        za = 4 * za * cos(orientation)
        w = ca * ra
        if w != 0:
            w = (w - za) / (w + w)
        icd = int(floor(ca * w + 0.5))
        ird = int(floor(ra * w + 0.5))

        # Draw the 4 quadrants
        rr_t, cc_t = _bezier_segment(ir0 + ird, ic0, ir0, ic0, ir0, ic0 + icd, 1-w)
        rr.extend(rr_t)
        cc.extend(cc_t)
        rr_t, cc_t = _bezier_segment(ir0 + ird, ic0, ir1, ic0, ir1, ic1 - icd, w)
        rr.extend(rr_t)
        cc.extend(cc_t)
        rr_t, cc_t = _bezier_segment(ir1 - ird, ic1, ir1, ic1, ir1, ic1 - icd, 1-w)
        rr.extend(rr_t)
        cc.extend(cc_t)
        rr_t, cc_t = _bezier_segment(ir1 - ird, ic1, ir0, ic1, ir0, ic0 + icd,  w)
        rr.extend(rr_t)
        cc.extend(cc_t)

    if shape is not None:
        return _coords_inside_image(np.array(rr, dtype=np.intp),
                                    np.array(cc, dtype=np.intp), shape)
    return np.array(rr, dtype=np.intp), np.array(cc, dtype=np.intp)


def _bezier_segment(Py_ssize_t r0, Py_ssize_t c0,
                    Py_ssize_t r1, Py_ssize_t c1,
                    Py_ssize_t r2, Py_ssize_t c2,
                    double weight):
    """Generate Bezier segment coordinates.

    Parameters
    ----------
    r0, c0 : int
        Coordinates of the first control point.
    r1, c1 : int
        Coordinates of the middle control point.
    r2, c2 : int
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
    cdef list cc = list()
    cdef list rr = list()

    # Steps
    cdef double sc = c2 - c1
    cdef double sr = r2 - r1

    cdef double d2c = c0 - c2
    cdef double d2r = r0 - r2
    cdef double d1c = c0 - c1
    cdef double d1r = r0 - r1
    cdef double rc = d1c * sr + d1r * sc
    cdef double cur = d1c * sr - d1r * sc
    cdef double err

    cdef bint test1, test2

    # If not a straight line
    if cur != 0 and weight > 0:
        if (sc * sc + sr * sr > d1c * d1c + d1r * d1r):
            # Swap point 0 and point 2
            # to start from the longer part
            c2 = c0
            c0 -= <Py_ssize_t>(d2c)
            r2 = r0
            r0 -= <Py_ssize_t>(d2r)
            cur = -cur
        d1c = 2 * (4 * weight * sc * d1c + d2c * d2c)
        d1r = 2 * (4 * weight * sr * d1r + d2r * d2r)
        # Set steps
        if c0 < c2:
            sc = 1
        else:
            sc = -1
        if r0 < r2:
            sr = 1
        else:
            sr = -1
        rc = -2 * sc * sr * (2 * weight * rc + d2c * d2r)

        if cur * sc * sr < 0:
            d1c = -d1c
            d1r = -d1r
            rc = -rc
            cur = -cur

        d2c = 4 * weight * (c1 - c0) * sr * cur + d1c / 2 + rc
        d2r = 4 * weight * (r0 - r1) * sc * cur + d1r / 2 + rc

        # Flat ellipse, algo fails
        if weight < 0.5 and (d2r > rc or d2c < rc):
            cur = (weight + 1) / 2
            weight = sqrt(weight)
            rc = 1. / (weight + 1)
            # Subdivide curve in half
            sc = floor((c0 + 2 * weight * c1 + c2) * rc * 0.5 + 0.5)
            sr = floor((r0 + 2 * weight * r1 + r2) * rc * 0.5 + 0.5)
            d2c = floor((weight * c1 + c0) * rc + 0.5)
            d2r = floor((r1 * weight + r0) * rc + 0.5)
            return _bezier_segment(r0, c0, <Py_ssize_t>(d2r), <Py_ssize_t>(d2c),
                                   <Py_ssize_t>(sr), <Py_ssize_t>(sc), cur)

        err = d2c + d2r - rc
        while d2r <= rc and d2c >= rc:
            cc.append(c0)
            rr.append(r0)
            if c0 == c2 and r0 == r2:
                # The job is done!
                return np.array(rr, dtype=np.intp), np.array(cc, dtype=np.intp)

            # Save boolean values
            test1 = 2 * err > d2r
            test2 = 2 * (err + d1r) < -d2r
            # Move (c0, r0) to the next position
            if 2 * err < d2c or test2:
                r0 += <Py_ssize_t>(sr)
                d2r += rc
                d2c += d1c
                err += d2c
            if 2 * err > d2c or test1:
                c0 += <Py_ssize_t>(sc)
                d2c += rc
                d2r += d1r
                err += d2r

    # Plot line
    cc_t, rr_t = _line(c0, r0, c2, r2)
    cc.extend(cc_t)
    rr.extend(rr_t)

    return np.array(rr, dtype=np.intp), np.array(cc, dtype=np.intp)


def _bezier_curve(Py_ssize_t r0, Py_ssize_t c0,
                  Py_ssize_t r1, Py_ssize_t c1,
                  Py_ssize_t r2, Py_ssize_t c2,
                  double weight, shape):
    """Generate Bezier curve coordinates.

    Parameters
    ----------
    r0, c0 : int
        Coordinates of the first control point.
    r1, c1 : int
        Coordinates of the middle control point.
    r2, c2 : int
        Coordinates of the last control point.
    weight : double
        Middle control point weight, it describes the line tension.
    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for curves that exceed the image
        size. If None, the full extent of the curve is used.

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
    cdef list cc = list()
    cdef list rr = list()

    cdef int vc, vr
    cdef double dc, dr, ww, t, q
    vc = c0 - 2 * c1 + c2
    vr = r0 - 2 * r1 + r2

    dc = c0 - c1
    dr = r0 - r1

    if dc * (c2 - c1) > 0:
        if dr * (r2 - r1):
            if abs(dc * vr) > abs(dr * vc):
                c0 = c2
                c2 = <Py_ssize_t>(dc + c1)
                r0 = r2
                r2 = <Py_ssize_t>(dr + r1)
        if (c0 == c2) or (weight == 1.):
            t = <double>(c0 - c1) / vc
        else:
            q = sqrt(4. * weight * weight * (c0 - c1) * (c2 - c1) + (c2 - c0) * floor(c2 - c0))
            if (c1 < c0):
                q = -q
            t = (2. * weight * (c0 - c1) - c0 + c2 + q) / (2. * (1. - weight) * (c2 - c0))

        q = 1. / (2. * t * (1. - t) * (weight - 1.) + 1.0)
        dc = (t * t * (c0 - 2. * weight * c1 + c2) + 2. * t * (weight * c1 - c0) + c0) * q
        dr = (t * t * (r0 - 2. * weight * r1 + r2) + 2. * t * (weight * r1 - r0) + r0) * q
        ww = t * (weight - 1.) + 1.
        ww *= ww * q
        weight = ((1. - t) * (weight - 1.) + 1.) * sqrt(q)
        vc = <int>(dc + 0.5)
        vr = <int>(dr + 0.5)
        dr = (dc - c0) * (r1 - r0) / (c1 - c0) + r0

        rr_t, cc_t = _bezier_segment(r0, c0, <int>(dr + 0.5), vc, vr, vc, ww)
        cc.extend(cc_t)
        rr.extend(rr_t)

        dr = (dc - c2) * (r1 - r2) / (c1 - c2) + r2
        r1 = <int>(dr + 0.5)
        c0 = c1 = vc
        r0 = vr

    if (r0 - r1) * floor(r2 - r1) > 0:
        if (r0 == r2) or (weight == 1):
            t = (r0 - r1) / (r0 - 2. * r1 + r2)
        else:
            q = sqrt(4. * weight * weight * (r0 - r1) * (r2 - r1) + (r2 - r0) * floor(r2 - r0))
            if r1 < r0:
                q = -q
            t = (2. * weight * (r0 - r1) - r0 + r2 + q) / (2. * (1. - weight) * (r2 - r0))
        q = 1. / (2. * t * (1. - t) * (weight - 1.) + 1.)
        dc = (t * t * (c0 - 2. * weight * c1 + c2) + 2. * t * (weight * c1 - c0) + c0) * q
        dr = (t * t * (r0 - 2. * weight * r1 + r2) + 2. * t * (weight * r1 - r0) + r0) * q
        ww = t * (weight - 1.) + 1.
        ww *= ww * q
        weight = ((1. - t) * (weight - 1.) + 1.) * sqrt(q)
        vc = <int>(dc + 0.5)
        vr = <int>(dr + 0.5)
        dc = (c1 - c0) * (dr - r0) / (r1 - r0) + c0

        rr_t, cc_t = _bezier_segment(r0, c0, vr, <int>(dc + 0.5), vr, vc, ww)
        cc.extend(cc_t)
        rr.extend(rr_t)

        dc = (c1 - c2) * (dr - r2) / (r1 - r2) + c2
        c1 = <int>(dc + 0.5)
        c0 = vc
        r0 = r1 = vr

    rr_t, cc_t = _bezier_segment(r0, c0, r1, c1, r2, c2, weight * weight)
    cc.extend(cc_t)
    rr.extend(rr_t)

    if shape is not None:
        return _coords_inside_image(np.array(rr, dtype=np.intp),
                                    np.array(cc, dtype=np.intp), shape)
    return np.array(rr, dtype=np.intp), np.array(cc, dtype=np.intp)
