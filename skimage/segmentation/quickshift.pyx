import numpy as np
cimport numpy as np

from itertools import product

from time import time

cdef extern from "math.h":
    double exp(double)


def quickshift(np.ndarray[dtype=np.float_t, ndim=3, mode="c"] image, sigma=5, tau=10, return_tree=False):
    """Computes quickshift clustering in RGB-(x,y) space.

    Parameters
    ----------
    image: ndarray, [width, height, channels]
        Input image
    sigma: float
        Width of Gaussian kernel used in smoothing the
        sample density. Higher means less clusters.
    tau: float
        Cut-off point for data distances.
        Higher means less clusters.
    return_tree: bool
        Whether to return the full segmentation hierarchy tree

    Returns
    -------
    segment_mask: ndarray, [width, height]
        Integer mask indicating segment labels.
    """

    # We compute the distances twice since otherwise
    # we might get crazy memory overhead (width * height * windowsize**2)

    # TODO do smoothing beforehand?
    # TODO manage borders somehow?

    # window size for neighboring pixels to consider
    if sigma < 1:
        raise ValueError("Sigma should be >= 1")
    cdef int w = int(2 * sigma)
    
    cdef int width = image.shape[0]
    cdef int height = image.shape[1]
    cdef int channels = image.shape[2]
    cdef float closest, dist
    cdef int x, y, xx, yy, x_, y_

    cdef np.ndarray[dtype=np.float_t, ndim=2] densities = np.zeros((width, height))
    start = time()
    # compute densities
    for x, y in product(xrange(width), xrange(height)):
        current_pixel = image[x, y, :]
        for xx, yy in product(xrange(-w / 2, w / 2 + 1), repeat=2):
            x_, y_ = x + xx, y + yy
            if 0 <= x_ < width and 0 <= y_ < height:
                dist = 0
                for c in xrange(channels):
                    dist += (current_pixel[c] - image[x_, y_, c])**2
                dist += (x - x_)**2 + (y - y_)**2
                densities[x, y] += float(exp(-dist / sigma))
    print("densities: %f" % (time() - start))

    # this will break ties that otherwise would give us headache

    densities += np.random.normal(scale=0.00001, size=(width, height))
    # default parent to self:
    cdef np.ndarray[dtype=np.int_t, ndim=2] parent = np.arange(width * height).reshape(width, height)
    cdef np.ndarray[dtype=np.float_t, ndim=2] dist_parent = np.zeros((width, height))
    start = time()
    # find nearest node with higher density
    for x, y in product(xrange(width), xrange(height)):
        current_density = densities[x, y]
        current_pixel = image[x, y, :]
        closest = np.inf
        for xx, yy in product(xrange(-w / 2, w / 2 + 1), repeat=2):
            x_, y_ = x + xx, y + yy
            if 0 <= x_ < width and 0 <= y_ < height:
                if densities[x_, y_] > current_density:
                    dist = 0
                    for c in xrange(channels):
                        dist += (current_pixel[c] - image[x_, y_, c])**2
                    dist += (x - x_)**2 + (y - y_)**2
                    if dist < closest:
                        closest = dist
                        parent[x, y] = x_ * width + y_
        dist_parent[x, y] = closest
    print("parents: %f" % (time() - start))

    start = time()
    dist_parent_flat = dist_parent.ravel()
    flat = parent.ravel()
    flat[dist_parent_flat > tau] = np.arange(width * height)[dist_parent_flat > tau]
    old = np.zeros_like(flat)
    while (old != flat).any():
        old = flat
        flat = flat[flat]
    print("rest: %f" % (time() - start))
    flat = flat.reshape(width, height)
    if return_tree:
        return flat, parent
    return flat
