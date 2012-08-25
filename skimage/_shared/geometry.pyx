#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


cdef inline unsigned char point_in_polygon(int nr_verts, double *xp, double *yp,
                                           double x, double y):
    """Test whether point lies inside a polygon.

    Parameters
    ----------
    nr_verts : int
        Number of vertices of polygon.
    xp, yp : double array
        Coordinates of polygon with length nr_verts.
    x, y : double
        Coordinates of point.
    """
    cdef int i
    cdef unsigned char c = 0
    cdef int j = nr_verts - 1
    for i in range(nr_verts):
        if (
            (((yp[i] <= y) and (y < yp[j])) or
            ((yp[j] <= y) and (y < yp[i])))
            and (x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i])
        ):
            c = not c
        j = i
    return c


cdef void points_in_polygon(int nr_verts, double *xp, double *yp,
                            int nr_points, double *x, double *y,
                            unsigned char *result):
    """Test whether points lie inside a polygon.

    Parameters
    ----------
    nr_verts : int
        Number of vertices of polygon.
    xp, yp : double array
        Coordinates of polygon with length nr_verts.
    nr_points : int
        Number of points to test.
    x, y : double array
        Coordinates of points.
    result : unsigned char array
        Test results for each point.
    """
    cdef int n
    for n in range(nr_points):
        result[n] = point_in_polygon(nr_verts, xp, yp, x[n], y[n])
