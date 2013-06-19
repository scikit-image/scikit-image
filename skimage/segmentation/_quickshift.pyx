#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
from scipy import ndimage
from itertools import product

cimport numpy as cnp
from libc.math cimport exp, sqrt

from ..util import img_as_float
from ..color import rgb2lab


def quickshift(image, ratio=1., float kernel_size=5, max_dist=10,
               return_tree=False, sigma=0, convert2lab=True, random_seed=None):
    """Segments image using quickshift clustering in Color-(x,y) space.

    Produces an oversegmentation of the image using the quickshift mode-seeking
    algorithm.

    Parameters
    ----------
    image : (width, height, channels) ndarray
        Input image.
    ratio : float, optional, between 0 and 1 (default 1).
        Balances color-space proximity and image-space proximity.
        Higher values give more weight to color-space.
    kernel_size : float, optional (default 5)
        Width of Gaussian kernel used in smoothing the
        sample density. Higher means fewer clusters.
    max_dist : float, optional (default 10)
        Cut-off point for data distances.
        Higher means fewer clusters.
    return_tree : bool, optional (default False)
        Whether to return the full segmentation hierarchy tree and distances.
    sigma : float, optional (default 0)
        Width for Gaussian smoothing as preprocessing. Zero means no smoothing.
    convert2lab : bool, optional (default True)
        Whether the input should be converted to Lab colorspace prior to
        segmentation. For this purpose, the input is assumed to be RGB.
    random_seed : None (default) or int, optional
        Random seed used for breaking ties.

    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.

    Notes
    -----
    The authors advocate to convert the image to Lab color space prior to
    segmentation, though this is not strictly necessary. For this to work, the
    image must be given in RGB format.

    References
    ----------
    .. [1] Quick shift and kernel methods for mode seeking,
           Vedaldi, A. and Soatto, S.
           European Conference on Computer Vision, 2008


    """
    image = img_as_float(np.atleast_3d(image))
    if convert2lab:
        if image.shape[2] != 3:
            ValueError("Only RGB images can be converted to Lab space.")
        image = rgb2lab(image)

    image = ndimage.gaussian_filter(img_as_float(image), [sigma, sigma, 0])
    cdef cnp.ndarray[dtype=cnp.float_t, ndim=3, mode="c"] image_c \
            = np.ascontiguousarray(image) * ratio

    random_state = np.random.RandomState(random_seed)

    # TODO join orphaned roots?
    # Some nodes might not have a point of higher density within the
    # search window. We could do a global search over these in the end.
    # Reference implementation doesn't do that, though, and it only has
    # an effect for very high max_dist.

    # window size for neighboring pixels to consider
    if kernel_size < 1:
        raise ValueError("Sigma should be >= 1")
    cdef int w = int(3 * kernel_size)

    cdef Py_ssize_t height = image_c.shape[0]
    cdef Py_ssize_t width = image_c.shape[1]
    cdef Py_ssize_t channels = image_c.shape[2]
    cdef double current_density, closest, dist

    cdef Py_ssize_t r, c, r_, c_, channel, r_min, c_min

    cdef cnp.float_t* image_p = <cnp.float_t*> image_c.data
    cdef cnp.float_t* current_pixel_p = image_p

    cdef cnp.ndarray[dtype=cnp.float_t, ndim=2] densities \
            = np.zeros((height, width))

    # compute densities
    for r in range(height):
        for c in range(width):
            r_min, r_max = max(r - w, 0), min(r + w + 1, height)
            c_min, c_max = max(c - w, 0), min(c + w + 1, width)
            for r_ in range(r_min, r_max):
                for c_ in range(c_min, c_max):
                    dist = 0
                    for channel in range(channels):
                        dist += (current_pixel_p[channel] -
                                 image_c[r_, c_, channel])**2
                    dist += (r - r_)**2 + (c - c_)**2
                    densities[r, c] += exp(-dist / (2 * kernel_size**2))
            current_pixel_p += channels

    # this will break ties that otherwise would give us headache
    densities += random_state.normal(scale=0.00001, size=(height, width))

    # default parent to self:
    cdef cnp.ndarray[dtype=cnp.int_t, ndim=2] parent \
            = np.arange(width * height).reshape(height, width)
    cdef cnp.ndarray[dtype=cnp.float_t, ndim=2] dist_parent \
            = np.zeros((height, width))

    # find nearest node with higher density
    current_pixel_p = image_p
    for r in range(height):
        for c in range(width):
            current_density = densities[r, c]
            closest = np.inf
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
                                     image_c[r_, c_, channel])**2
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
