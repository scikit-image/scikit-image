# -*- python -*-

cimport numpy as np
import numpy as np

cdef extern from "_pnpoly.h":
     void npnpoly(int nr_verts, double *xp, double *yp,
                  int nr_points, double *x, double *y,
                  unsigned char *result)


def points_inside_poly(points, verts):
     """Test whether points lie inside a polygon.

     Parameters
     ----------
     points : (N, 2) array
         Input points, ``(x, y)``.
     verts : (M, 2) array
         Vertices of the polygon, sorted either clockwise or anti-clockwise.

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

