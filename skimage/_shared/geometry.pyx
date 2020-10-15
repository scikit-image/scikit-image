#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp
cnp.import_array()


cdef unsigned char point_in_polygon(np_floats[::1] xp, np_floats[::1] yp,
                                    np_floats x, np_floats y,
                                    np_floats eps) nogil:
    """Test relative point position to a polygone.

    Parameters
    ----------
    nr_verts : int
        Number of vertices of polygon.
    xp, yp : np_floats array
        Coordinates of polygon with length nr_verts.
    x, y : np_floats
        Coordinates of point.
    eps : np_floats
        Inclusion tolerence.

    Returns
    -------
    c : unsigned char
        Point relative position to the polygone O: outside, 1: inside,
        2: vertex; 3: edge.

    References
    ----------
    .. [1] O'Rourke (1998), "Computational Geometry in C",
           Second Edition, Cambridge Unversity Press, Chapter 7
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t nr_verts = xp.shape[0]
    cdef Py_ssize_t j = nr_verts - 1
    cdef np_floats x0, x1, y0, y1, c
    cdef unsigned int l_cross, r_cross

    for i in range(nr_verts):
        x0 = xp[i] - x
        y0 = yp[i] - y

        if (-eps < x0 < eps) and (-eps < y0 < eps):
            # it's a vertex
            return VERTEX

        x1 = xp[j] - x
        y1 = yp[j] - y
        c = (x0 * y1 - x1 * y0) / (y1 - y0)

        if (# straddles the x-component of the ray
            y0 > 0 != y1 > 0
            # crosses the ray if strictly positive intersection
            and c > 0):
            r_cross += 1

        if (
            # straddles the x-component of the ray when reversed
            y0 < 0 != y1 < 0
            # crosses the ray if strictly negative intersection
            and c < 0
        ):
            l_cross += 1

        j = i

    if (r_cross + l_cross) & 1:
        # on edge if left and right crossings not of same parity
        return EDGE

    if r_cross & 1:
        # inside if odd number of crossings
        return INSIDE

    # outside if even number of crossings
    return OUTSIDE


cdef void points_in_polygon(np_floats[::1] xp, np_floats[::1] yp,
                            np_floats[::1] x, np_floats[::1] y,
                            unsigned char[::1] result,
                            np_floats eps) nogil:
    """Test whether points lie inside a polygon.

    Parameters
    ----------
    nr_verts : int
        Number of vertices of polygon.
    xp, yp : np_floats array
        Coordinates of polygon with length nr_verts.
    nr_points : int
        Number of points to test.
    x, y : np_floats array
        Coordinates of points.
    result : unsigned char array
        Test results for each point.
    """
    cdef Py_ssize_t n
    cdef Py_ssize_t nr_points = x.shape[0]
    for n in range(nr_points):
        result[n] = point_in_polygon(xp, yp, x[n], y[n], eps)
