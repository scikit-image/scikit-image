#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp

from .._shared.fused_numerics cimport np_floats, np_ints

from libc.math cimport exp, sqrt, ceil
from libc.float cimport DBL_MAX

cnp.import_array()


def _quickshift_cython(np_floats[:, :, ::1] image, np_floats kernel_size,
                       np_floats max_dist, bint return_tree, int random_seed,
                       bool subset, np_ints[:, :] subset_idxs):
    """
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
    random_seed : {None, int, `numpy.random.Generator`}, optional
        If `random_seed` is None the `numpy.random.Generator` singleton
        is used.
        If `random_seed` is an int, a new ``Generator`` instance is used,
        seeded with `random_seed`.
        If `random_seed` is already a ``Generator`` instance then that instance
        is used.

        Random seed used for breaking ties.

    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.
    """

    random_state = np.random.default_rng(random_seed)

    if np_floats is cnp.float64_t:
        dtype = np.float64
    else:
        dtype = np.float32

    # TODO join orphaned roots?
    # Some nodes might not have a point of higher density within the
    # search window. We could do a global search over these in the end.
    # Reference implementation doesn't do that, though, and it only has
    # an effect for very high max_dist.

    # window size for neighboring pixels to consider
    cdef np_floats inv_kernel_size_sqr = -0.5 / (kernel_size * kernel_size)
    cdef int kernel_width = <int>ceil(3 * kernel_size)

    cdef Py_ssize_t height = image.shape[0]
    cdef Py_ssize_t width = image.shape[1]
    cdef Py_ssize_t channels = image.shape[2]

    cdef np_floats[:, ::1] densities = np.zeros((height, width), dtype=dtype)

    cdef np_floats current_density, closest, dist, t
    cdef Py_ssize_t r, c, r_, c_, channel, r_min, r_max, c_min, c_max
    cdef np_floats* current_pixel_ptr

    # this will break ties that otherwise would give us headache
    densities += random_state.normal(
        scale=0.00001, size=(height, width)
    ).astype(dtype, copy=False)

    # default parent to self
    cdef Py_ssize_t[:, ::1] parent = \
        np.arange(width * height, dtype=np.intp).reshape(height, width)
    cdef np_floats[:, ::1] dist_parent = np.zeros((height, width), dtype=dtype)

    # compute densities
    with nogil:
        current_pixel_ptr = &image[0, 0, 0]
        for r in range(height):
            # Check if row is in the image subset
            if subset and r not in subset_idxs:
                current_pixel_ptr += channels * width
                continue

            r_min = max(r - kernel_width, 0)
            r_max = min(r + kernel_width + 1, height)
            for c in range(width):
                # Check if (r, c) is in the image subset
                if subset and not np.any(np.equal(np.array([[r, c]]), subset_idxs).all(1)):
                    current_pixel_ptr += channels
                    continue

                c_min = max(c - kernel_width, 0)
                c_max = min(c + kernel_width + 1, width)
                for r_ in range(r_min, r_max):
                    for c_ in range(c_min, c_max):
                        # Check if (r_, c_) is in the image subset
                        if subset and not np.any(np.equal(np.array([[r_, c_]]), subset_idxs).all(1)):
                            continue

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
            # Check if row is in the image subset
            if subset and r not in subset_idxs:
                current_pixel_ptr += channels * width
                parent[r, :] = np.empty((width)).fill(-1)
                continue

            r_min = max(r - kernel_width, 0)
            r_max = min(r + kernel_width + 1, height)
            for c in range(width):
                # Check if (r, c) is in the image subset
                if subset and not np.any(np.equal(np.array([[r, c]]), subset_idxs).all(1)):
                    current_pixel_ptr += channels
                    parent[r, c] = -1
                    continue

                current_density = densities[r, c]
                closest = DBL_MAX
                c_min = max(c - kernel_width, 0)
                c_max = min(c + kernel_width + 1, width)
                for r_ in range(r_min, r_max):
                    for c_ in range(c_min, c_max):
                        # Check if (r_, c_) is in the image subset
                        if subset and not np.any(np.equal(np.array([[r_, c_]]), subset_idxs).all(1)):
                            continue

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
                # note that closest will never be zero, as dist being zero could only happen 
                # when finding the distance from a pixel to itself. But the density at a pixel will never
                # be greater than itself
                # So if dist_parent[r, c] == 0, we know (r, c) is not in the image subset
                dist_parent[r, c] = sqrt(closest)
                current_pixel_ptr += channels

    dist_parent_flat = np.array(dist_parent).ravel()
    parent_flat = np.array(parent).ravel()

    # remove parents with distance > max_dist
    too_far = dist_parent_flat > max_dist
    parent_flat[too_far] = np.arange(width * height)[too_far]
    old = np.zeros_like(parent_flat)

    # flatten forest (mark each pixel with root of corresponding tree)
    # set parent_flat[k] to parent_flat[parent_flat[k]]
    # so then if this doesn't change, we know we've found the root parent of a cluster
    while (old != parent_flat).any():
        old = parent_flat
        parent_flat = parent_flat[parent_flat]

    parent_flat = np.unique(parent_flat, return_inverse=True)[1]
    parent_flat = parent_flat.reshape(height, width)

    if return_tree:
        return parent_flat, parent, dist_parent
    return parent_flat
