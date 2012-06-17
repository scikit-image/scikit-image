import numpy as np
cimport numpy as np
from collections import defaultdict
import scipy


#from ..util import img_as_float
#from ..color import rgb2grey
#from skimage.morphology.ccomp cimport find_root, join_trees

DTYPE = np.int
ctypedef np.int_t DTYPE_t

cdef DTYPE_t find_root(np.int_t *forest, np.int_t n):
    """Find the root of node n.

    """
    cdef np.int_t root = n
    while (forest[root] < root):
        root = forest[root]
    return root

cdef set_root(np.int_t *forest, np.int_t n, np.int_t root):
    """
    Set all nodes on a path to point to new_root.

    """
    cdef np.int_t j
    while (forest[n] < n):
        j = forest[n]
        forest[n] = root
        n = j

    forest[n] = root


cdef join_trees(np.int_t *forest, np.int_t n, np.int_t m):
    """Join two trees containing nodes n and m.

    """
    cdef np.int_t root = find_root(forest, n)
    cdef np.int_t root_m

    if (n != m):
        root_m = find_root(forest, m)

        if (root > root_m):
            root = root_m

        set_root(forest, n, root)
        set_root(forest, m, root)


def felzenszwalb_segmentation_grey(image, scale=200, sigma=0.8):
    """Computes Felsenszwalb's efficient graph based segmentation for a single channel.

    Produces an oversegmentation of a 2d image using a fast, minimum spanning
    tree based clustering on the image grid.  The parameter ``scale`` sets an
    observation level. Higher scale means less and larger segments. ``sigma``
    is the diameter of a Gaussian kernel, used for smoothing the image prior to
    segmentation.

    The number of produced segments as well as their size can only be
    controlled indirectly through ``scale``. Segment size within an image can
    vary greatly depending on local contrast.

    Parameters
    ----------
    image: ndarray, [width, height]
        Input image

    scale: float
        Free parameter. Higher means larger clusters.
        For 0-255 data, hundereds are good.

    sigma: float
        Width of Gaussian kernel used in preprocessing.

    Returns
    -------
    segment_mask: ndarray, [width, height]
        Integer mask indicating segment labels.
    """
    if image.ndim != 2:
        raise ValueError("This algorithm works only on single-channel 2d images."
                "Got image of shape %s" % str(image.shape))
    scale = float(scale)
    image = scipy.ndimage.gaussian_filter(image, sigma=sigma)

    # compute edge weights in 8 connectivity:
    right_cost = np.abs((image[1:, :] - image[:-1, :]))
    down_cost = np.abs((image[:, 1:] - image[:, :-1]))
    dright_cost = np.abs((image[1:, 1:] - image[:-1, :-1]))
    uright_cost = np.abs((image[1:, :-1] - image[:-1, 1:]))
    cdef np.ndarray[np.float_t, ndim=1] costs = np.hstack([right_cost.ravel(), down_cost.ravel(),
        dright_cost.ravel(), uright_cost.ravel()]).astype(np.float)
    # compute edges between pixels:
    width, height = image.shape[:2]
    cdef np.ndarray[np.int_t, ndim=2] segments = np.arange(width * height).reshape(width, height)
    right_edges = np.c_[segments[1:, :].ravel(), segments[:-1, :].ravel()]
    down_edges = np.c_[segments[:, 1:].ravel(), segments[:, :-1].ravel()]
    dright_edges = np.c_[segments[1:, 1:].ravel(), segments[:-1, :-1].ravel()]
    uright_edges = np.c_[segments[:-1, 1:].ravel(), segments[1:, :-1].ravel()]
    cdef np.ndarray[np.int_t, ndim=2] edges = np.vstack([right_edges, down_edges, dright_edges, uright_edges])
    # initialize data structures for segment size
    # and inner cost, then start greedy iteration over edges.
    edge_queue = np.argsort(costs)
    edges = np.ascontiguousarray(edges[edge_queue])
    costs = np.ascontiguousarray(costs[edge_queue])
    cdef np.int_t *segments_p = <np.int_t*>segments.data
    cdef np.int_t *edges_p = <np.int_t*>edges.data
    cdef np.float_t *costs_p = <np.float_t*>costs.data
    cdef np.ndarray[np.int_t, ndim=1] segment_size = np.ones(width * height, dtype=np.int)
    # inner cost of segments
    cdef np.ndarray[np.float_t, ndim=1] cint = np.zeros(width * height)
    cdef int seg0, seg1, seg_new
    cdef float cost, inner_cost0, inner_cost1
    # set costs_p back one. we increase it before we use it
    # since we might continue before that.
    costs_p -= 1
    for e in xrange(costs.size):
        seg0 = find_root(segments_p, edges_p[0])
        seg1 = find_root(segments_p, edges_p[1])
        edges_p += 2
        costs_p += 1
        if seg0 == seg1:
            continue
        inner_cost0 = cint[seg0] + scale / segment_size[seg0]
        inner_cost1 = cint[seg1] + scale / segment_size[seg1]
        if costs_p[0] < min(inner_cost0, inner_cost1):
            # update size and cost
            join_trees(segments_p, seg0, seg1)
            seg_new = find_root(segments_p, seg0)
            segment_size[seg_new] = segment_size[seg0] + segment_size[seg1]
            cint[seg_new] = costs_p[0]
    
    # unravel the union find tree
    flat = segments.ravel()
    old = np.zeros_like(flat)
    while (old != flat).any():
        old = flat
        flat = flat[flat]
    return flat.reshape((width, height))
