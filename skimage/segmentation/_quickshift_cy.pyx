#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp

from libc.math cimport exp, sqrt, ceil
from libc.float cimport DBL_MAX


def _quickshift_cython(double[:, :, ::1] image, double kernel_size,
                       double max_dist, bint return_tree, int random_seed,
                       bint full_search):
    """Segments image using quickshift clustering in Color-(x,y) space.
    Produces an oversegmentation of the image using the quickshift mode-seeking
    algorithm.
    Parameters
    ----------
    image : (width, height, channels) ndarray
        Input image.
    kernel_size : float
        Width of Gaussian kernel used in smoothing the
        sample density. Higher means fewer clusters.
    max_dist : float
        Cut-off point for data distances.
        Higher means fewer clusters.
    return_tree : bool
        Whether to return the full segmentation hierarchy tree and distances.
    random_seed : int
        Random seed used for breaking ties.
    full_search : bool
        Wether to extend search to always find the nearest node with higher
        density. Will return a single tree if max_dist is large enough.
    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.
    """

    random_state = np.random.RandomState(random_seed)

    # window size for neighboring pixels to consider
    cdef double inv_kernel_size_sqr = -0.5 / (kernel_size * kernel_size)
    cdef int kernel_width = <int>ceil(3 * kernel_size)

    cdef Py_ssize_t height = image.shape[0]
    cdef Py_ssize_t width = image.shape[1]
    cdef Py_ssize_t channels = image.shape[2]

    cdef double[:, ::1] densities = np.zeros((height, width), dtype=np.double)

    cdef double current_density, closest, dist, t
    cdef Py_ssize_t r, c, r_, c_, channel, r_min, r_max, c_min, c_max
    cdef Py_ssize_t c_min_old, c_max_old, r_min_old, r_max_old
    cdef int window_size
    cdef double* current_pixel_ptr

    # this will break ties that otherwise would give us headache
    densities += random_state.normal(scale=0.00001, size=(height, width))
    # default parent to self
    cdef Py_ssize_t[:, ::1] parent = \
        np.arange(width * height, dtype=np.intp).reshape(height, width)
    cdef double[:, ::1] dist_parent = np.zeros((height, width),
                                               dtype=np.double)

    # compute densities
    with nogil:
        current_pixel_ptr = &image[0, 0, 0]
        for r in range(height):
            r_min = max(r - kernel_width, 0)
            r_max = min(r + kernel_width + 1, height)
            for c in range(width):
                c_min = max(c - kernel_width, 0)
                c_max = min(c + kernel_width + 1, width)
                for r_ in range(r_min, r_max):
                    for c_ in range(c_min, c_max):
                        dist = 0
                        for channel in range(channels):
                            t = (current_pixel_ptr[channel] -
                                 image[r_, c_, channel])
                            dist += t * t
                        t = r - r_
                        dist += t * t
                        t = c - c_
                        dist += t * t
                        densities[r, c] += exp(dist * inv_kernel_size_sqr)
                current_pixel_ptr += channels

        # find nearest node with higher density
        current_pixel_ptr = &image[0, 0, 0]
        for r in range(height):
            for c in range(width):
                current_density = densities[r, c]
                closest = DBL_MAX
                window_size = kernel_width
                c_min = max(c - window_size, 0)
                c_max = min(c + window_size + 1, width)
                r_min = max(r - window_size, 0)
                r_max = min(r + window_size + 1, height)
                c_min_old = 0
                c_max_old = 0
                r_min_old = 0
                r_max_old = 0
                # increase search window until you find a parent
                # or until you have searched all the image
                while closest == DBL_MAX and \
                    not(c_min_old == 0 and c_max_old == width and
                        r_min_old == 0 and r_max_old == height):
                    for r_ in range(r_min, r_max):
                        for c_ in range(c_min, c_max):
                            # no need to check the previous search window again
                            if not (c_min_old <= c_ < c_max_old and
                                    r_min_old <= r_ < r_max_old) and \
                                    densities[r_, c_] > current_density:
                                dist = 0
                                # We compute the distances twice since
                                # otherwise we get crazy memory overhead
                                # (width * height * windowsize**2)
                                for channel in range(channels):
                                    t = (current_pixel_ptr[channel] -
                                         image[r_, c_, channel])
                                    dist += t * t
                                t = r - r_
                                dist += t * t
                                t = c - c_
                                dist += t * t
                                if dist < closest:
                                    closest = dist
                                    parent[r, c] = r_ * width + c_
                    if not full_search:
                        break
                    c_min_old = c_min
                    c_max_old = c_max
                    r_min_old = r_min
                    r_max_old = r_max
                    window_size = window_size+kernel_width
                    c_min = max(c - window_size, 0)
                    c_max = min(c + window_size + 1, width)
                    r_min = max(r - window_size, 0)
                    r_max = min(r + window_size + 1, height)
                dist_parent[r, c] = sqrt(closest)
                current_pixel_ptr += channels

    dist_parent_flat = np.array(dist_parent).ravel()
    parent_flat = np.array(parent).ravel()

    # remove parents with distance > max_dist
    too_far = dist_parent_flat > max_dist
    parent_flat[too_far] = np.arange(width * height)[too_far]
    old = np.zeros_like(parent_flat)

    # flatten forest (mark each pixel with root of corresponding tree)
    while (old != parent_flat).any():
        old = parent_flat
        parent_flat = parent_flat[parent_flat]

    parent_flat = np.unique(parent_flat, return_inverse=True)[1]
    parent_flat = parent_flat.reshape(height, width)

    if return_tree:
        return parent_flat, parent, dist_parent
    return parent_flat
