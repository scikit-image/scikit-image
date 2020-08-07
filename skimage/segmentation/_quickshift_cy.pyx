#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp

from libc.math cimport exp, sqrt, ceil
from libc.float cimport DBL_MAX

cnp.import_array()


def _quickshift_cython(double[:, :, ::1] image, double kernel_size,
                       double max_dist, bint return_tree, int random_seed):
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

    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.
    """

    random_state = np.random.RandomState(random_seed)

    # TODO join orphaned roots?
    # Some nodes might not have a point of higher density within the
    # search window. We could do a global search over these in the end.
    # Reference implementation doesn't do that, though, and it only has
    # an effect for very high max_dist.

    # window size for neighboring pixels to consider
    cdef double inv_kernel_size_sqr = -0.5 / (kernel_size * kernel_size)
    cdef int kernel_width = <int>ceil(3 * kernel_size)

    cdef Py_ssize_t height = image.shape[0]
    cdef Py_ssize_t width = image.shape[1]
    cdef Py_ssize_t channels = image.shape[2]

    cdef double[:, ::1] densities = np.zeros((height, width), dtype=np.double)

    cdef double current_density, closest, dist, t
    cdef Py_ssize_t r, c, r_, c_, channel, r_min, r_max, c_min, c_max
    cdef double* current_pixel_ptr

    # this will break ties that otherwise would give us headache
    densities += random_state.normal(scale=0.00001, size=(height, width))
    # default parent to self
    cdef Py_ssize_t[:, ::1] parent = \
        np.arange(width * height, dtype=np.intp).reshape(height, width)
    cdef double[:, ::1] dist_parent = np.zeros((height, width), dtype=np.double)

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
            r_min = max(r - kernel_width, 0)
            r_max = min(r + kernel_width + 1, height)
            for c in range(width):
                current_density = densities[r, c]
                closest = DBL_MAX
                c_min = max(c - kernel_width, 0)
                c_max = min(c + kernel_width + 1, width)
                for r_ in range(r_min, r_max):
                    for c_ in range(c_min, c_max):
                        if densities[r_, c_] > current_density:
                            dist = 0
                            # We compute the distances twice since otherwise
                            # we get crazy memory overhead
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
