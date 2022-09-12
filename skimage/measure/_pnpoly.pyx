#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp
from .._shared.geometry cimport point_in_polygon, points_in_polygon

cnp.import_array()


def _grid_points_in_poly(shape, verts, return_labels=False):
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
    return_labels: bool, optional
        If ``True``, a label mask will also be returned along with the default
        boolean mask. The possible labels are: O - outside, 1 - inside,
        2 - vertex, 3 - edge.

    See Also
    --------
    points_in_poly

    Returns
    -------
    mask : (M, N) ndarray of bool
        True where the grid falls inside the polygon.
    labels: (M, N) ndarray of labels (integers)
        Labels array, with pixels having a label between 0 and 3.
        This is only returned if `return_labels` is set to True.

    """
    verts = np.asarray(verts)

    cdef cnp.float64_t[::1] vx = verts[:, 0].astype(np.float64)
    cdef cnp.float64_t[::1] vy = verts[:, 1].astype(np.float64)

    cdef Py_ssize_t M = shape[0]
    cdef Py_ssize_t N = shape[1]
    cdef Py_ssize_t m, n

    cdef cnp.ndarray[dtype=cnp.uint8_t, ndim=2, mode="c"] out = \
         np.zeros((M, N), dtype=np.uint8)

    with nogil:
        for m in range(M):
            for n in range(N):
                out[m, n] = point_in_polygon(vx, vy, m, n)

    # In case the consumer of this function would like to transform
    # the labels array into a mask manually, we shall return the raw labels.
    if return_labels:
        return out.view(bool), out

    return out.view(bool)


def _points_in_poly(points, verts):
    """Test whether points lie inside a polygon.

    Parameters
    ----------
    points : (N, 2) array
        Input points, ``(x, y)``.
    verts : (M, 2) array
        Vertices of the polygon, sorted either clockwise or anti-clockwise.
        The first point may (but does not need to be) duplicated.

    See Also
    --------
    grid_points_in_poly

    Returns
    -------
    mask : (N,) array of bool
        True if corresponding point is inside the polygon.

    """
    points = np.asarray(points)
    verts = np.asarray(verts)

    cdef cnp.float64_t[::1] x = points[:, 0].astype(np.float64)
    cdef cnp.float64_t[::1] y = points[:, 1].astype(np.float64)

    cdef cnp.float64_t[::1] vx = verts[:, 0].astype(np.float64)
    cdef cnp.float64_t[::1] vy = verts[:, 1].astype(np.float64)

    cdef unsigned char[::1] out = np.zeros(x.shape[0], dtype=bool)

    points_in_polygon(vx, vy, x, y, out)

    return np.asarray(out)
