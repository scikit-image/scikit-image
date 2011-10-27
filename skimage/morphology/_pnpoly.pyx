# -*- python -*-

cimport numpy as np
import numpy as np

cdef extern from "_pnpoly.h":
     int pnpoly(int nr_verts, double *xp, double *yp,
                double x, double y)

     void npnpoly(int nr_verts, double *xp, double *yp,
                  int nr_points, double *x, double *y,
                  unsigned char *result)


def grid_points_inside_poly(shape, verts):
    """Test whether points on a specified grid are inside a polygon.

    For each ``(r, c)`` coordinate on a grid, i.e. ``(0, 0)``, ``(0, 1)`` etc.,
    test whether that point lies inside a polygon.

    Parameters
    ----------
    shape : tuple (M, N)
        Shape of the grid.
    verts : (V, 2) array
        Specify the V vertices of the polygon, sorted either clockwise
        or anti-clockwise.  The first point may (but does not need to be)
        duplicated.

    Returns
    -------
    mask : (M, N) ndarray of bool
        True where the grid falls inside the polygon.

    """
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] vx, vy
    verts = np.asarray(verts)

    vx = verts[:, 0].astype(np.double)
    vy = verts[:, 1].astype(np.double)
    cdef int V = vx.shape[0]

    cdef int M = shape[0]
    cdef int N = shape[1]
    cdef int m, n

    cdef np.ndarray[dtype=np.uint8_t, ndim=2, mode="c"] out = \
         np.zeros((M, N), dtype=np.uint8)

    for m in range(M):
        for n in range(N):
            out[m, n] = pnpoly(V, <double*>vx.data, <double*>vy.data, m, n)

    return out.view(bool)
    

def points_inside_poly(points, verts):
     """Test whether points lie inside a polygon.

     Parameters
     ----------
     points : (N, 2) array
         Input points, ``(x, y)``.
     verts : (M, 2) array
         Vertices of the polygon, sorted either clockwise or anti-clockwise.
         The first point may (but does not need to be) duplicated.

     Returns
     -------
     mask : (N,) array of bool
         True if corresponding point is inside the polygon.

     """
     cdef np.ndarray[np.double_t, ndim=1, mode="c"] x, y, vx, vy

     points = np.asarray(points)
     verts = np.asarray(verts)

     x = points[:, 0].astype(np.double)
     y = points[:, 1].astype(np.double)

     vx = verts[:, 0].astype(np.double)
     vy = verts[:, 1].astype(np.double)

     cdef np.ndarray[np.uint8_t, ndim=1] out = \
          np.zeros(x.shape[0], dtype=np.uint8)
     
     npnpoly(vx.shape[0], <double*>vx.data, <double*>vy.data,
             x.shape[0], <double*>x.data, <double*>y.data,
             <unsigned char*>out.data)

     return out.astype(bool)

