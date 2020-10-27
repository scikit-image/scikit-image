#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp
cnp.import_array()


cdef unsigned char point_in_polygon(np_floats[::1] xp, np_floats[::1] yp,
                                    np_floats x, np_floats y) nogil:
    """Test relative point position to a polygon.

    Parameters
    ----------
    xp, yp : np_floats array
        Coordinates of polygon with length nr_verts.
    x, y : np_floats
        Coordinates of point.

    Returns
    -------
    c : unsigned char
        Point relative position to the polygon O: outside, 1: inside,
        2: vertex; 3: edge.

    References
    ----------
    .. [1] O'Rourke (1998), "Computational Geometry in C",
           Second Edition, Cambridge Unversity Press, Chapter 7
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t nr_verts = xp.shape[0]
    cdef np_floats x0, x1, y0, y1, eps
    cdef unsigned int l_cross = 0, r_cross = 0

    # Tolerance for vertices labelling
    eps = 1e-12

    # Initialization the loop
    x1 = xp[nr_verts - 1] - x
    y1 = yp[nr_verts - 1] - y

    # For each edge e=(i-1, i), see if it crosses ray
    for i in range(nr_verts):
        x0 = xp[i] - x
        y0 = yp[i] - y

        if (-eps < x0 < eps) and (-eps < y0 < eps):
            # it is a vertex with an eps tolerance
            return VERTEX

        # if e straddles the x-axis
        if ((y0 > 0) != (y1 > 0)):
            # check if it crosses the ray
            if ((x0 * y1 - x1 * y0) / (y1 - y0)) > 0:
                r_cross += 1
        # if reversed e straddles the x-axis
        if ((y0 < 0) != (y1 < 0)):
            # check if it crosses the ray
            if ((x0 * y1 - x1 * y0) / (y1 - y0)) < 0:
                l_cross += 1

        x1 = x0
        y1 = y0

    if (r_cross & 1) != (l_cross & 1):
        # on edge if left and right crossings not of same parity
        return EDGE

    if r_cross & 1:
        # inside if odd number of crossings
        return INSIDE

    # outside if even number of crossings
    return OUTSIDE


cdef void points_in_polygon(np_floats[::1] xp, np_floats[::1] yp,
                            np_floats[::1] x, np_floats[::1] y,
                            unsigned char[::1] result) nogil:
    """Test whether points lie inside a polygon.

    Parameters
    ----------
    xp, yp : np_floats array
        Coordinates of polygon with length nr_verts.
    x, y : np_floats array
        Coordinates of points.
    result : unsigned char array
        Test results for each point.
    """
    cdef Py_ssize_t n
    cdef Py_ssize_t nr_points = x.shape[0]
    for n in range(nr_points):
        result[n] = point_in_polygon(xp, yp, x[n], y[n])
