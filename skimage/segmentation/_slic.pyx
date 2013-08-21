#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
from scipy import ndimage

from skimage.util import img_as_float, regular_grid
from skimage.color import rgb2lab, gray2rgb


def _slic_cython(double[:, :, :, ::1] image_zyx,
                 Py_ssize_t[:, :, ::1] nearest_mean,
                 double[:, :, ::1] distance,
                 double[:, ::1] means,
                 Py_ssize_t max_iter, Py_ssize_t n_segments):
    """Helper function for SLIC segmentation.

    Parameters
    ----------
    image_zyx : 4D array of double, shape (Z, Y, X, 6)
        The image with embedded coordinates, that is, `image_zyx[i, j, k]` is
        `array([i, j, k, r, g, b])` or `array([i, j, k, L, a, b])`, depending
        on the colorspace.
    nearest_mean : 3D array of int, shape (Z, Y, X)
        The (initially empty) label field.
    distance : 3D array of double, shape (Z, Y, X)
        The (initially infinity) array of distances to the nearest centroid.
    means : 2D array of double, shape (n_segments, 6)
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
    # approximate grid size for desired n_segments
    cdef Py_ssize_t step_z, step_y, step_x
    slices = regular_grid((depth, height, width), n_segments)
    step_z, step_y, step_x = [int(s.step) for s in slices]

    n_means = means.shape[0]
    cdef Py_ssize_t i, k, x, y, z, x_min, x_max, y_min, y_max, z_min, z_max, \
            changes
    cdef double dist_mean
    cdef double tmp
    for i in range(max_iter):
        changes = 0
        distance[:, :, :] = np.inf
        # assign pixels to means
        for k in range(n_means):
            # compute windows:
            z_min = int(max(means[k, 0] - 2 * step_z, 0))
            z_max = int(min(means[k, 0] + 2 * step_z, depth))
            y_min = int(max(means[k, 1] - 2 * step_y, 0))
            y_max = int(min(means[k, 1] + 2 * step_y, height))
            x_min = int(max(means[k, 2] - 2 * step_x, 0))
            x_max = int(min(means[k, 2] + 2 * step_x, width))
            for z in range(z_min, z_max):
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        dist_mean = 0
                        for c in range(6):
                            # you would think the compiler can optimize the
                            # squaring itself. mine can't (with O2)
                            tmp = image_zyx[z, y, x, c] - means[k, c]
                            dist_mean += tmp * tmp
                        if distance[z, y, x] > dist_mean:
                            nearest_mean[z, y, x] = k
                            distance[z, y, x] = dist_mean
                            changes = 1
        if changes == 0:
            break
        # recompute means:
        nearest_mean_ravel = np.asarray(nearest_mean).ravel()
        means_list = []
        for j in range(6):
            image_zyx_ravel = (
                        np.ascontiguousarray(image_zyx[:, :, :, j]).ravel())
            means_list.append(np.bincount(nearest_mean_ravel,
                                          image_zyx_ravel))
        in_mean = np.bincount(nearest_mean_ravel)
        in_mean[in_mean == 0] = 1
        means = (np.vstack(means_list) / in_mean).T.copy("C")
    return np.ascontiguousarray(nearest_mean)
