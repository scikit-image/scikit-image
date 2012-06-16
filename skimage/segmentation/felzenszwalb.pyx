import numpy as np
cimport numpy as np
from collections import defaultdict
import scipy

#from ..util import img_as_float
#from ..color import rgb2grey
from skimage.morphology.ccomp cimport find_root, join_trees

from IPython.core.debugger import Tracer
tracer = Tracer()


def felzenszwalb_segmentation(image, k, sigma=0.8):
    k = float(k)
    #image = img_as_float(image)
    #image = rgb2grey(image)
    image = image[:, :, 0]
    image = scipy.ndimage.gaussian_filter(image, sigma=sigma)

    # compute edge weights in 8 connectivity:
    #right_cost = np.sum((image[1:, :, :] - image[:-1, :, :]) ** 2, axis=2)
    #down_cost = np.sum((image[:, 1:, :] - image[:, :-1, :]) ** 2, axis=2)
    right_cost = np.abs((image[1:, :] - image[:-1, :]))
    down_cost = np.abs((image[:, 1:] - image[:, :-1]))
    dright_cost = np.abs((image[1:, 1:] - image[:-1, :-1]))
    uright_cost = np.abs((image[1:, :-1] - image[:-1, 1:]))
    costs = np.hstack([right_cost.ravel(), down_cost.ravel(),
        dright_cost.ravel(), uright_cost.ravel()])
    # compute edges between pixels:
    width, height = image.shape[:2]
    indices = np.arange(width * height).reshape(width, height)
    right_edges = np.c_[indices[1:, :].ravel(), indices[:-1, :].ravel()]
    down_edges = np.c_[indices[:, 1:].ravel(), indices[:, :-1].ravel()]
    dright_edges = np.c_[indices[1:, 1:].ravel(), indices[:-1, :-1].ravel()]
    uright_edges = np.c_[indices[:-1, 1:].ravel(), indices[1:, :-1].ravel()]
    edges = np.vstack([right_edges, down_edges, dright_edges, uright_edges])
    # initialize data structures for segment size
    # and inner cost, then start greedy iteration over edges.
    edge_queue = np.argsort(costs)
    cdef np.ndarray[np.int_t, ndim=2] segments = indices.reshape(width, height)
    cdef np.int_t *segments_p = <np.int_t*>segments.data
    cdef np.int_t seg_new
    segment_size = defaultdict(lambda: 1)
    # inner cost of segments
    cint = defaultdict(lambda: 0)
    for edge, cost in zip(edges[edge_queue], costs[edge_queue]):
        seg0 = find_root(segments_p, edge[0])
        seg1 = find_root(segments_p, edge[1])
        if seg0 == seg1:
            continue
        inner_cost0 = cint[seg0] + k / segment_size[seg0]
        inner_cost1 = cint[seg1] + k / segment_size[seg1]
        if cost < min(inner_cost0, inner_cost1):
            # update size and cost
            join_trees(segments_p, seg0, seg1)
            seg_new = find_root(segments_p, seg0)
            segment_size[seg_new] = segment_size[seg0] + segment_size[seg1]
            cint[seg_new] = cost
    return segments
