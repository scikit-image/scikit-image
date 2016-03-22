#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp

from libc.math cimport exp, sqrt
from libc.float cimport DBL_MAX


def _quickshift_cython(cnp.ndarray[double, ndim=3, mode="c"] image,
                       float kernel_size,
                       float max_dist,
                       bint return_tree,
                       int random_seed):
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
    cdef float kernel_size_sq = kernel_size**2
    cdef int w = np.ceil(3 * kernel_size)

    cdef Py_ssize_t height = image.shape[0]
    cdef Py_ssize_t width = image.shape[1]
    cdef Py_ssize_t channels = image.shape[2]
    cdef double current_density, closest, dist

    cdef Py_ssize_t r, c, r_, c_, channel, r_min, c_min

    cdef cnp.float_t* image_p = <cnp.float_t*> image.data
    cdef cnp.float_t* current_pixel_p

    cdef cnp.ndarray[dtype=cnp.float_t, ndim=2] densities \
            = np.zeros((height, width))

    # compute densities
    with nogil:
        current_pixel_p = image_p
        for r in range(height):
            for c in range(width):
                r_min, r_max = max(r - w, 0), min(r + w + 1, height)
                c_min, c_max = max(c - w, 0), min(c + w + 1, width)
                for r_ in range(r_min, r_max):
                    for c_ in range(c_min, c_max):
                        dist = 0
                        for channel in range(channels):
                            dist += (current_pixel_p[channel] -
                                     image[r_, c_, channel])**2
                        dist += (r - r_)**2 + (c - c_)**2
                        densities[r, c] += exp(-dist / (2 * kernel_size_sq))
                current_pixel_p += channels

    # this will break ties that otherwise would give us headache
    densities += random_state.normal(scale=0.00001, size=(height, width))

    # default parent to self
    cdef cnp.ndarray[dtype=cnp.int_t, ndim=2] parent \
            = np.arange(width * height).reshape(height, width)
    cdef cnp.ndarray[dtype=cnp.float_t, ndim=2] dist_parent \
            = np.zeros((height, width))

    # find nearest node with higher density
    with nogil:
        current_pixel_p = image_p
        for r in range(height):
            for c in range(width):
                current_density = densities[r, c]
                closest = DBL_MAX
                r_min, r_max = max(r - w, 0), min(r + w + 1, height)
                c_min, c_max = max(c - w, 0), min(c + w + 1, width)
                for r_ in range(r_min, r_max):
                    for c_ in range(c_min, c_max):
                        if densities[r_, c_] > current_density:
                            dist = 0
                            # We compute the distances twice since otherwise
                            # we get crazy memory overhead
                            # (width * height * windowsize**2)
                            for channel in range(channels):
                                dist += (current_pixel_p[channel] -
                                         image[r_, c_, channel])**2
                            dist += (r - r_)**2 + (c - c_)**2
                            if dist < closest:
                                closest = dist
                                parent[r, c] = r_ * width + c_
                dist_parent[r, c] = sqrt(closest)
                current_pixel_p += channels

    dist_parent_flat = dist_parent.ravel()
    flat = parent.ravel()
    # remove parents with distance > max_dist
    too_far = dist_parent_flat > max_dist
    flat[too_far] = np.arange(width * height)[too_far]
    old = np.zeros_like(flat)
    # flatten forest (mark each pixel with root of corresponding tree)
    while (old != flat).any():
        old = flat
        flat = flat[flat]
    flat = np.unique(flat, return_inverse=True)[1]
    flat = flat.reshape(height, width)
    if return_tree:
        return flat, parent, dist_parent
    return flat
