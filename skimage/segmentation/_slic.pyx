#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.float cimport DBL_MAX

import numpy as np
cimport numpy as cnp

from skimage.util import regular_grid


def _slic_cython(double[:, :, :, ::1] image_zyx,
                 Py_ssize_t[:, :, ::1] nearest_mean,
                 double[:, :, ::1] distance,
                 double[:, ::1] clusters,
                 Py_ssize_t max_iter, Py_ssize_t n_segments):
    """Helper function for SLIC segmentation.

    Parameters
    ----------
    image_zyx : 4D array of double, shape (Z, Y, X, C)
        The image with embedded coordinates, that is, `image_zyx[i, j, k]` is
        `array([i, j, k, c])`, depending
        on the colorspace.
    nearest_mean : 3D array of int, shape (Z, Y, X)
        The (initially empty) label field.
    distance : 3D array of double, shape (Z, Y, X)
        The (initially infinity) array of distances to the nearest centroid.
    clusters : 2D array of double, shape (n_segments, 6)
        The centroids obtained by SLIC.
    max_iter : int
        The maximum number of k-means iterations.
    n_segments : int
        The approximate/desired number of segments.

    Returns
    -------
    nearest_mean : 3D array of int, shape (Z, Y, X)
        The label field/superpixels found by SLIC.
    """

    # initialize on grid:
    cdef Py_ssize_t depth, height, width
    depth, height, width = (image_zyx.shape[0], image_zyx.shape[1],
                            image_zyx.shape[2])

    cdef Py_ssize_t n_features = clusters.shape[1]
    cdef Py_ssize_t n_clusters = clusters.shape[0]

    # approximate grid size for desired n_segments
    cdef Py_ssize_t step_z, step_y, step_x
    slices = regular_grid((depth, height, width), n_segments)
    step_z, step_y, step_x = [int(s.step) for s in slices]

    cdef Py_ssize_t i, k, x, y, z, x_min, x_max, y_min, y_max, z_min, z_max, \
            changes
    cdef double dist_mean
    cdef double tmp

    cdef Py_ssize_t[:] n_cluster_elems = np.zeros(n_clusters, dtype=np.intp)

    for i in range(max_iter):
        changes = 0
        distance[:, :, :] = DBL_MAX

        # assign pixels to clusters
        for k in range(n_clusters):
            # compute windows:
            z_min = int(max(clusters[k, 0] - 2 * step_z, 0))
            z_max = int(min(clusters[k, 0] + 2 * step_z, depth))
            y_min = int(max(clusters[k, 1] - 2 * step_y, 0))
            y_max = int(min(clusters[k, 1] + 2 * step_y, height))
            x_min = int(max(clusters[k, 2] - 2 * step_x, 0))
            x_max = int(min(clusters[k, 2] + 2 * step_x, width))
            for z in range(z_min, z_max):
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        dist_mean = 0
                        for c in range(n_features):
                            # you would think the compiler can optimize the
                            # squaring itself. mine can't (with O2)
                            tmp = image_zyx[z, y, x, c] - clusters[k, c]
                            dist_mean += tmp * tmp
                        if distance[z, y, x] > dist_mean:
                            nearest_mean[z, y, x] = k
                            distance[z, y, x] = dist_mean
                            changes = 1
        if changes == 0:
            break

        # recompute clusters

        # sum features for all clusters
        n_cluster_elems[:] = 0
        clusters[:, :] = 0
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    k = nearest_mean[z, y, x]
                    n_cluster_elems[k] += 1
                    for c in range(n_features):
                        clusters[k, c] += image_zyx[z, y, x, c]

        # divide by number of elements per cluster to obtain mean
        for k in range(n_clusters):
            for c in range(n_features):
                clusters[k, c] /= n_cluster_elems[k]

    return np.asarray(nearest_mean)
