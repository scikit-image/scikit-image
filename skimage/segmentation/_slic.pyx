#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.float cimport DBL_MAX

import numpy as np
cimport numpy as cnp

from skimage.util import regular_grid


def _slic_cython(double[:, :, :, ::1] image_zyx,
                 double[:, ::1] clusters,
                 Py_ssize_t max_iter):
    """Helper function for SLIC segmentation.

    Parameters
    ----------
    image_zyx : 4D array of double, shape (Z, Y, X, C)
        The input image.
    clusters : 2D array of double, shape (N, 3 + C)
        The initial centroids obtained by SLIC as [Z, Y, X, C...].
    max_iter : int
        The maximum number of k-means iterations.

    Returns
    -------
    nearest_clusters : 3D array of int, shape (Z, Y, X)
        The label field/superpixels found by SLIC.
    """

    # initialize on grid
    cdef Py_ssize_t depth, height, width
    depth, height, width = (image_zyx.shape[0], image_zyx.shape[1],
                            image_zyx.shape[2])

    cdef Py_ssize_t n_clusters = clusters.shape[0]
    # number of features [X, Y, Z, ...]
    cdef Py_ssize_t n_features = clusters.shape[1]

    # approximate grid size for desired n_segments
    cdef Py_ssize_t step_z, step_y, step_x
    slices = regular_grid((depth, height, width), n_clusters)
    step_z, step_y, step_x = [int(s.step) for s in slices]

    cdef Py_ssize_t[:, :, ::1] nearest_clusters \
        = np.empty((depth, height, width), dtype=np.intp)
    cdef double[:, :, ::1] distance \
        = np.empty((depth, height, width), dtype=np.double)

    cdef Py_ssize_t i, c, k, x, y, z, x_min, x_max, y_min, y_max, z_min, \
                    z_max
    cdef double dist_mean

    cdef char change

    cdef double cx, cy, cz, dy, dz

    cdef Py_ssize_t[:] n_cluster_elems = np.zeros(n_clusters, dtype=np.intp)

    for i in range(max_iter):
        change = 0
        distance[:, :, :] = DBL_MAX

        # assign pixels to clusters
        for k in range(n_clusters):

            # cluster coordinate centers
            cz = clusters[k, 0]
            cy = clusters[k, 1]
            cx = clusters[k, 2]

            # compute windows
            z_min = <Py_ssize_t>max(cz - 2 * step_z, 0)
            z_max = <Py_ssize_t>min(cz + 2 * step_z + 1, depth)
            y_min = <Py_ssize_t>max(cy - 2 * step_y, 0)
            y_max = <Py_ssize_t>min(cy + 2 * step_y + 1, height)
            x_min = <Py_ssize_t>max(cx - 2 * step_x, 0)
            x_max = <Py_ssize_t>min(cx + 2 * step_x + 1, width)

            for z in range(z_min, z_max):
                dz = (cz - z) ** 2
                for y in range(y_min, y_max):
                    dy = (cy - y) ** 2
                    for x in range(x_min, x_max):
                        dist_mean = dz + dy + (cx - x) ** 2
                        for c in range(3, n_features):
                            dist_mean += (image_zyx[z, y, x, c - 3]
                                          - clusters[k, c]) ** 2
                        if distance[z, y, x] > dist_mean:
                            nearest_clusters[z, y, x] = k
                            distance[z, y, x] = dist_mean
                            change = 1

        # stop if no pixel changed its cluster
        if change == 0:
            break

        # recompute clusters

        # sum features for all clusters
        n_cluster_elems[:] = 0
        clusters[:, :] = 0
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    k = nearest_clusters[z, y, x]
                    n_cluster_elems[k] += 1
                    clusters[k, 0] += z
                    clusters[k, 1] += y
                    clusters[k, 2] += x
                    for c in range(3, n_features):
                        clusters[k, c] += image_zyx[z, y, x, c - 3]

        # divide by number of elements per cluster to obtain mean
        for k in range(n_clusters):
            for c in range(n_features):
                clusters[k, c] /= n_cluster_elems[k]

    return np.asarray(nearest_clusters)
