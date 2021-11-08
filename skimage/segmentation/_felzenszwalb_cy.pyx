#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp

from .._shared.filters import gaussian
from .._shared.utils import warn
from ..measure._ccomp cimport find_root, join_trees
from ..util import img_as_float64

cnp.import_array()


def _felzenszwalb_cython(image, double scale=1, sigma=0.8,
                         Py_ssize_t min_size=20):
    """Felzenszwalb's efficient graph based segmentation for
    single or multiple channels.

    Produces an oversegmentation of a single or multi-channel image
    using a fast, minimum spanning tree based clustering on the image grid.
    The number of produced segments as well as their size can only be
    controlled indirectly through ``scale``. Segment size within an image can
    vary greatly depending on local contrast.

    Parameters
    ----------
    image : (N, M, C) ndarray
        Input image.
    scale : float, optional (default 1)
        Sets the obervation level. Higher means larger clusters.
    sigma : float, optional (default 0.8)
        Width of Gaussian smoothing kernel used in preprocessing.
        Larger sigma gives smother segment boundaries.
    min_size : int, optional (default 20)
        Minimum component size. Enforced using postprocessing.

    Returns
    -------
    segment_mask : (N, M) ndarray
        Integer mask indicating segment labels.
    """

    if image.shape[2] > 3:
        warn(RuntimeWarning(
            "Got image with third dimension of %s. This image "
            "will be interpreted as a multichannel 2d image, "
            "which may not be intended." % str(image.shape[2])),
            stacklevel=3)

    image = img_as_float64(image)

    # rescale scale to behave like in reference implementation
    scale = float(scale) / 255.
    image = gaussian(image, sigma=[sigma, sigma, 0], mode='reflect',
                     channel_axis=-1)
    height, width = image.shape[:2]

    # compute edge weights in 8 connectivity:
    down_cost = np.sqrt(np.sum((image[1:, :, :] - image[:height-1, :, :]) *
    	                         (image[1:, :, :] - image[:height-1, :, :]), axis=-1))
    right_cost = np.sqrt(np.sum((image[:, 1:, :] - image[:, :width-1, :]) *
    	                          (image[:, 1:, :] - image[:, :width-1, :]), axis=-1))
    dright_cost = np.sqrt(np.sum((image[1:, 1:, :] - image[:height-1, :width-1, :]) *
	                               (image[1:, 1:, :] - image[:height-1, :width-1, :]), axis=-1))
    uright_cost = np.sqrt(np.sum((image[1:, :width-1, :] - image[:height-1, 1:, :]) *
    	                           (image[1:, :width-1, :] - image[:height-1, 1:, :]), axis=-1))
    cdef cnp.ndarray[cnp.float_t, ndim=1] costs = np.hstack([
    	right_cost.ravel(), down_cost.ravel(), dright_cost.ravel(),
        uright_cost.ravel()]).astype(float)

    # compute edges between pixels:
    cdef cnp.ndarray[cnp.intp_t, ndim=2] segments \
            = np.arange(width * height, dtype=np.intp).reshape(height, width)
    down_edges  = np.c_[segments[1:, :].ravel(), segments[:height-1, :].ravel()]
    right_edges = np.c_[segments[:, 1:].ravel(), segments[:, :width-1].ravel()]
    dright_edges = np.c_[segments[1:, 1:].ravel(), segments[:height-1, :width-1].ravel()]
    uright_edges = np.c_[segments[:height-1, 1:].ravel(), segments[1:, :width-1].ravel()]
    cdef cnp.ndarray[cnp.intp_t, ndim=2] edges \
            = np.vstack([right_edges, down_edges, dright_edges, uright_edges])

    # initialize data structures for segment size
    # and inner cost, then start greedy iteration over edges.
    edge_queue = np.argsort(costs)
    edges = np.ascontiguousarray(edges[edge_queue])
    costs = np.ascontiguousarray(costs[edge_queue])
    cdef cnp.intp_t *segments_p = <cnp.intp_t*>segments.data
    cdef cnp.intp_t *edges_p = <cnp.intp_t*>edges.data
    cdef cnp.float_t *costs_p = <cnp.float_t*>costs.data
    cdef cnp.ndarray[cnp.intp_t, ndim=1] segment_size \
            = np.ones(width * height, dtype=np.intp)

    # inner cost of segments
    cdef cnp.ndarray[cnp.float_t, ndim=1] cint = np.zeros(width * height)
    cdef cnp.intp_t seg0, seg1, seg_new, e
    cdef float cost, inner_cost0, inner_cost1
    cdef Py_ssize_t num_costs = costs.size

    with nogil:
        # set costs_p back one. we increase it before we use it
        # since we might continue before that.
        costs_p -= 1
        for e in range(num_costs):
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

        # postprocessing to remove small segments
        edges_p = <cnp.intp_t*>edges.data
        for e in range(num_costs):
            seg0 = find_root(segments_p, edges_p[0])
            seg1 = find_root(segments_p, edges_p[1])
            edges_p += 2
            if seg0 == seg1:
                continue
            if segment_size[seg0] < min_size or segment_size[seg1] < min_size:
                join_trees(segments_p, seg0, seg1)
                seg_new = find_root(segments_p, seg0)
                segment_size[seg_new] = segment_size[seg0] + segment_size[seg1]

    # unravel the union find tree
    flat = segments.ravel()
    old = np.zeros_like(flat)
    while (old != flat).any():
        old = flat
        flat = flat[flat]
    flat = np.unique(flat, return_inverse=True)[1]
    return flat.reshape((height, width))
