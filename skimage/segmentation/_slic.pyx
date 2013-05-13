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


def _slic_cython(cnp.ndarray[dtype=cnp.float_t, ndim=4] image_zyx,
                 cnp.ndarray[dtype=cnp.intp_t, ndim=3] nearest_mean,
                 cnp.ndarray[dtype=cnp.float_t, ndim=3] distance,
                 cnp.ndarray[dtype=cnp.float_t, ndim=2] means,
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

    cdef cnp.float_t* current_mean
    cdef cnp.float_t* mean_entry
    n_means = means.shape[0]
    cdef Py_ssize_t i, k, x, y, z, x_min, x_max, y_min, y_max, z_min, z_max, \
            changes
    cdef double dist_mean

    cdef cnp.float_t* image_p = <cnp.float_t*> image_zyx.data
    cdef cnp.float_t* distance_p = <cnp.float_t*> distance.data
    cdef cnp.float_t* current_distance
    cdef cnp.float_t* current_pixel
    cdef double tmp
    for i in range(max_iter):
        distance.fill(np.inf)
        changes = 0
        current_mean = <cnp.float_t*> means.data
        # assign pixels to means
        for k in range(n_means):
            # compute windows:
            z_min = int(max(current_mean[0] - 2 * step_z, 0))
            z_max = int(min(current_mean[0] + 2 * step_z, depth))
            y_min = int(max(current_mean[1] - 2 * step_y, 0))
            y_max = int(min(current_mean[1] + 2 * step_y, height))
            x_min = int(max(current_mean[2] - 2 * step_x, 0))
            x_max = int(min(current_mean[2] + 2 * step_x, width))
            for z in range(z_min, z_max):
                for y in range(y_min, y_max):
                    current_pixel = \
                            &image_p[6 * ((z * height + y) * width + x_min)]
                    current_distance = \
                            &distance_p[(z * height + y) * width + x_min]
                    for x in range(x_min, x_max):
                        mean_entry = current_mean
                        dist_mean = 0
                        for c in range(6):
                            # you would think the compiler can optimize the
                            # squaring itself. mine can't (with O2)
                            tmp = current_pixel[0] - mean_entry[0]
                            dist_mean += tmp * tmp
                            current_pixel += 1
                            mean_entry += 1
                        # some precision issue here. Doesnt work if testing ">"
                        if current_distance[0] - dist_mean > 1e-10:
                            nearest_mean[z, y, x] = k
                            current_distance[0] = dist_mean
                            changes += 1
                        current_distance += 1
            current_mean += 6
        if changes == 0:
            break
        # recompute means:
        means_list = [np.bincount(nearest_mean.ravel(),
                      image_zyx[:, :, :, j].ravel()) for j in range(6)]
        in_mean = np.bincount(nearest_mean.ravel())
        in_mean[in_mean == 0] = 1
        means = (np.vstack(means_list) / in_mean).T.copy("C")
    return nearest_mean
