#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import collections as coll
import numpy as np
from time import time
from scipy import ndimage

cimport numpy as cnp

from ..util import img_as_float, regular_grid
from ..color import rgb2lab, gray2rgb


def _slic_cython(double[:, :, :, ::1] image_zyx,
                 int[:, :, ::1] nearest_mean,
                 double[:, :, ::1] distance,
                 double[:, ::1] means,
                 float ratio, int max_iter, int n_segments):
    """Helper function for SLIC segmentation."""

    # initialize on grid:
    cdef Py_ssize_t depth, height, width
    shape = image_zyx.shape
    depth, height, width = shape[0], shape[1], shape[2]
    # approximate grid size for desired n_segments
    cdef Py_ssize_t step_z, step_y, step_x
    grid_z, grid_y, grid_x = np.mgrid[:depth, :height, :width]
    slices = regular_grid((depth, height, width), n_segments)
    step_z, step_y, step_x = [int(s.step) for s in slices]

    n_means = means.shape[0]
    cdef Py_ssize_t i, k, x, y, z, x_min, x_max, y_min, y_max, z_min, z_max, \
            changes
    cdef double dist_mean

    cdef double tmp
    for i in range(max_iter):
        distance.fill(np.inf)
        changes = 0
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
                        # some precision issue here. Doesnt work if testing ">"
                        if distance[z, y, x] - dist_mean > 1e-10:
                            nearest_mean[z, y, x] = k
                            distance[z, y, x] = dist_mean
                            changes = 1
        if changes == 0:
            break
        # recompute means:
        means_list = [np.bincount(nearest_mean.ravel(),
                      image_zyx[:, :, :, j].ravel()) for j in range(6)]
        in_mean = np.bincount(nearest_mean.ravel())
        in_mean[in_mean == 0] = 1
        means = (np.vstack(means_list) / in_mean).T.copy("C")
    return nearest_mean
