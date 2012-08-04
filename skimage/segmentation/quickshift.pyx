import numpy as np
cimport numpy as np
cimport cython

from itertools import product

from ..util import img_as_float
from ..color import rgb2lab


cdef extern from "math.h":
    double exp(double)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def quickshift(image, ratio=1., kernel_size=5, max_dist=10, return_tree=False, convert2lab=True, random_seed=None):
    """Segments image using quickshift clustering in Color-(x,y) space.

    Produces an oversegmentation of the image using the quickshift mode-seeking algorithm.

    Parameters
    ----------
    image: (width, height, channels) ndarray 
        Input image
    ratio: float, between 0 and 1.
        Balances color-space proximity and image-space proximity.
        Higher values give more weight to color-space.
    kernel_size: float
        Width of Gaussian kernel used in smoothing the
        sample density. Higher means less clusters.
    max_dist: float
        Cut-off point for data distances.
        Higher means less clusters.
    return_tree: bool
        Whether to return the full segmentation hierarchy tree
    convert2lab: bool
        Whether the input should be converted to Lab colorspace prior to segmentation.
        For this purpose, the input is assumed to be RGB.
    random_seed: None or int
        Random seed used for breaking ties

    Returns
    -------
    segment_mask: ndarray, [width, height]
        Integer mask indicating segment labels.

    Notes
    -----
    The authors advocate to convert the image to Lab color space prior to segmentation, though
    this is not strictly necessary. For this to work, the image must be given in RGB format.

    References
    ----------
    .. [1] Quick shift and kernel methods for mode seeking, Vedaldi, A. and Soatto, S.
           European Conference on Computer Vision, 2008


    """
    image = img_as_float(np.atleast_3d(image))
    if convert2lab:
        if image.shape[2] != 3:
            ValueError("Only RGB images can be converted to Lab space.")
        image = rgb2lab(image)

    cdef np.ndarray[dtype=np.float_t, ndim=3, mode="c"] image_c = np.ascontiguousarray(image) * ratio

    if random_seed is None:
        random_state = np.random.RandomState()
    else:
        random_state = np.random.RandomState(random_seed)

    # We compute the distances twice since otherwise
    # we get crazy memory overhead (width * height * windowsize**2)

    # TODO join orphant roots?
    # Some nodes might not have a point of higher density within the
    # search window. We could do a global search over these in the end.
    # Reference implementation doesn't do that, though, and it only has
    # an effect for very high max_dist.

    # window size for neighboring pixels to consider
    if kernel_size < 1:
        raise ValueError("Sigma should be >= 1")
    cdef int w = int(3 * kernel_size)

    cdef int width = image_c.shape[0]
    cdef int height = image_c.shape[1]
    cdef int channels = image_c.shape[2]
    cdef float closest, dist
    cdef int x, y, x_, y_

    cdef np.float_t* image_p = <np.float_t*> image_c.data
    cdef np.float_t* current_pixel_p = image_p

    cdef np.ndarray[dtype=np.float_t, ndim=2] densities = np.zeros((width, height))
    # compute densities
    for x, y in product(xrange(width), xrange(height)):
        x_min, x_max = max(x - w, 0), min(x + w + 1, width)
        y_min, y_max = max(y - w, 0), min(y + w + 1, height)
        for x_, y_ in product(xrange(x_min, x_max), xrange(y_min, y_max)):
            dist = 0
            for c in xrange(channels):
                dist += (current_pixel_p[c] - image_c[x_, y_, c])**2
            dist += (x - x_)**2 + (y - y_)**2
            densities[x, y] += exp(-dist / (2 * kernel_size**2))
        current_pixel_p += channels

    # this will break ties that otherwise would give us headache
    densities += random_state.normal(scale=0.00001, size=(width, height))

    # default parent to self:
    cdef np.ndarray[dtype=np.int_t, ndim=2] parent = np.arange(width * height).reshape(width, height)
    cdef np.ndarray[dtype=np.float_t, ndim=2] dist_parent = np.zeros((width, height))
    # find nearest node with higher density
    current_pixel_p = image_p
    for x, y in product(xrange(width), xrange(height)):
        current_density = densities[x, y]
        closest = np.inf
        x_min, x_max = max(x - w, 0), min(x + w + 1, width)
        y_min, y_max = max(y - w, 0), min(y + w + 1, height)
        for x_, y_ in product(xrange(x_min, x_max), xrange(y_min, y_max)):
            if densities[x_, y_] > current_density:
                dist = 0
                for c in xrange(channels):
                    dist += (current_pixel_p[c] - image_c[x_, y_, c])**2
                dist += (x - x_)**2 + (y - y_)**2
                if dist < closest:
                    closest = dist
                    parent[x, y] = x_ * width + y_
        dist_parent[x, y] = np.sqrt(closest)
        current_pixel_p += channels

    dist_parent_flat = dist_parent.ravel()
    flat = parent.ravel()
    # remove parents with distance > max_dist
    flat[dist_parent_flat > max_dist] = np.arange(width * height)[dist_parent_flat > max_dist]
    old = np.zeros_like(flat)
    # flatten forest (mark each pixel with root of corresponding tree)
    while (old != flat).any():
        old = flat
        flat = flat[flat]
    flat = np.unique(flat, return_inverse=True)[1]
    flat = flat.reshape(width, height)
    if return_tree:
        return flat, parent, dist_parent
    return flat
