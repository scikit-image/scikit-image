try:
    import networkx as nx
except ImportError:
    msg = "Graph functions require networkx, which is not installed"

    class nx:
        class Graph:
            def __init__(self, *args, **kwargs):
                raise ImportError(msg)
    import warnings
    warnings.warn(msg)
import numpy as np
from scipy.ndimage import filters
from scipy import ndimage as nd
import math
from .. import draw, measure, segmentation


def min_weight(graph, src, dst, n):
    """Callback to handle merging nodes by choosing minimum weight.

    Returns either the weight between (`src`, `n`) or (`dst`, `n`)
    in `graph` or the minumum of the two when both exist.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The verices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    weight : float
        The weight between (`src`, `n`) or (`dst`, `n`) in `graph` or the
        minumum of the two when both exist.

    """

    # cover the cases where n only has edge to either `src` or `dst`
    default = {'weight': np.inf}
    w1 = graph[n].get(src, default)['weight']
    w2 = graph[n].get(dst, default)['weight']
    return min(w1, w2)


class RAG(nx.Graph):

    """
    The Region Adjacency Graph (RAG) of an image, subclasses
    `networx.Graph <http://networkx.github.io/documentation/latest/reference/classes.graph.html>`_
    """

    def merge_nodes(self, src, dst, weight_func=min_weight, extra_arguments=[],
                    extra_keywords={}):
        """Merge node `src` into `dst`.

        The new combined node is adjacent to all the neighbors of `src`
        and `dst`. `weight_func` is called to decide the weight of edges
        incident on the new node.

        Parameters
        ----------
        src, dst : int
            Nodes to be merged.
        weight_func : callable, optional
            Function to decide edge weight of edges incident on the new node.
            For each neighbor `n` for `src and `dst`, `weight_func` will be
            called as follows: `weight_func(src, dst, n, *extra_arguments,
            **extra_keywords)`. `src`, `dst` and `n` are IDs of vertices in the
            RAG object which is in turn a subclass of
            `networkx.Graph`.
        extra_arguments : sequence, optional
            The sequence of extra positional arguments passed to
            `weight_func`.
        extra_keywords : dictionary, optional
            The dict of keyword arguments passed to the `weight_func`.

        """
        src_nbrs = set(self.neighbors(src))
        dst_nbrs = set(self.neighbors(dst))
        neighbors = (src_nbrs & dst_nbrs) - set([src, dst])

        for neighbor in neighbors:
            w = weight_func(self, src, dst, neighbor, *extra_arguments,
                            **extra_keywords)
            self.add_edge(neighbor, dst, weight=w)

        self.node[dst]['labels'] += self.node[src]['labels']
        self.remove_node(src)


def _add_edge_filter(values, graph):
    """Create edge in `g` between the first element of `values` and the rest.

    Add an edge between the first element in `values` and
    all other elements of `values` in the graph `g`. `values[0]`
    is expected to be the central value of the footprint used.

    Parameters
    ----------
    values : array
        The array to process.
    graph : RAG
        The graph to add edges in.

    Returns
    -------
    0 : int
        Always returns 0. The return value is required so that `generic_filter`
        can put it in the output array.

    """
    values = values.astype(int)
    current = values[0]
    for value in values[1:]:
        graph.add_edge(current, value)

    return 0


def rag_mean_color(image, labels, connectivity=2, mode='distance',
                   sigma=255.0):
    """Compute the Region Adjacency Graph using mean colors.

    Given an image and its initial segmentation, this method constructs the
    corresponsing Region Adjacency Graph (RAG). Each node in the RAG
    represents a set of pixels within `image` with the same label in `labels`.
    The weight between two adjacent regions represents how similar or
    dissimilar two regions are depending on the `mode` parameter.

    Parameters
    ----------
    image : ndarray, shape(M, N, [..., P,] 3)
        Input image.
    labels : ndarray, shape(M, N, [..., P,])
        The labelled image. This should have one dimension less than
        `image`. If `image` has dimensions `(M, N, 3)` `labels` should have
         dimensions `(M, N)`.
    connectivity : int, optional
        Pixels with a squared distance less than `connectivity` from each other
        are considered adjacent. It can range from 1 to `labels.ndim`. Its
        behavior is the same as `connectivity` parameter in
        `scipy.ndimage.filters.generate_binary_structure`.
    mode : {'distance', 'similarity'}, optional
        The strategy to assign edge weights.

            'distance' : The weight between two adjacent regions is the
            :math:`|c_1 - c_2|`, where :math:`c_1` and :math:`c_2` are the mean
            colors of the two regions. It represents the Euclidean distance in
            their average color.

            'similarity' : The weight between two adjacent is
            :math:`e^{-d^2/sigma}` where :math:`d=|c_1 - c_2|`, where
            :math:`c_1` and :math:`c_2` are the mean colors of the two regions.
            It represents how similar two regions are.
    sigma : float, optional
        Used for computation when `mode` is "similarity". It governs how
        close to each other two colors should be, for their corresponding edge
        weight to be significant. A very large value of `sigma` could make
        any two colors behave as though they were similar.

    Returns
    -------
    out : RAG
        The region adjacency graph.

    Examples
    --------
    >>> from skimage import data, graph, segmentation
    >>> img = data.lena()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_mean_color(img, labels)

    References
    ----------
    .. [1] Alain Tremeau and Philippe Colantoni
           "Regions Adjacency Graph Applied To Color Image Segmentation"
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.11.5274

    """
    graph = RAG()

    # The footprint is constructed in such a way that the first
    # element in the array being passed to _add_edge_filter is
    # the central value.
    fp = nd.generate_binary_structure(labels.ndim, connectivity)
    for d in range(fp.ndim):
        fp = fp.swapaxes(0, d)
        fp[0, ...] = 0
        fp = fp.swapaxes(0, d)

    # For example
    # if labels.ndim = 2 and connectivity = 1
    # fp = [[0,0,0],
    #       [0,1,1],
    #       [0,1,0]]
    #
    # if labels.ndim = 2 and connectivity = 2
    # fp = [[0,0,0],
    #       [0,1,1],
    #       [0,1,1]]

    filters.generic_filter(
        labels,
        function=_add_edge_filter,
        footprint=fp,
        mode='nearest',
        output=np.zeros(labels.shape, dtype=np.uint8),
        extra_arguments=(graph,))

    for n in graph:
        graph.node[n].update({'labels': [n],
                              'pixel count': 0,
                              'total color': np.array([0, 0, 0],
                                                      dtype=np.double)})

    for index in np.ndindex(labels.shape):
        current = labels[index]
        graph.node[current]['pixel count'] += 1
        graph.node[current]['total color'] += image[index]

    for n in graph:
        graph.node[n]['mean color'] = (graph.node[n]['total color'] /
                                       graph.node[n]['pixel count'])

    for x, y, d in graph.edges_iter(data=True):
        diff = graph.node[x]['mean color'] - graph.node[y]['mean color']
        diff = np.linalg.norm(diff)
        if mode == 'similarity':
            d['weight'] = math.e ** (-(diff ** 2) / sigma)
        elif mode == 'distance':
            d['weight'] = diff
        else:
            raise ValueError("The mode '%s' is not recognised" % mode)

    return graph



def rag_draw(labels, rag, img, border_color = (0,0,0), node_color = (1,1,0), 
             low_color = (0,1,0), high_color=None, thresh=np.inf ):
    rag = rag.copy()
    rag_labels = labels.copy()
    out = img.copy()
    
    low_color = np.array(low_color)
    
    if not high_color is None :
        high_color = np.array(high_color)

    # Handling the case where one node has multiple labels
    offset = 1
    for n, d in rag.nodes_iter(data=True):
        for l in d['labels'] :
            rag_labels[labels == l] = offset
        offset += 1

    regions = measure.regionprops(rag_labels)
    for region in regions:
        # Because we kept the offset as 1
        rag.node[region['label'] - 1]['centroid'] = region['centroid']

    if not border_color is None :
        out = segmentation.mark_boundaries(out, rag_labels, color = border_color)

    if not high_color is None :
        max_weight = max([d['weight'] for x, y, d in rag.edges_iter(data=True) if d['weight'] < thresh ])
        min_weight = min([d['weight'] for x, y, d in rag.edges_iter(data=True) if d['weight'] < thresh ])
    
    for n1,n2,data in rag.edges_iter(data=True):

        if data['weight'] >=  thresh :
            continue
        r1, c1 = map(int, rag.node[n1]['centroid'])
        r2, c2 = map(int, rag.node[n2]['centroid'])

        line  = draw.line(r1, c1, r2, c2)
                
        if not high_color is None:
            norm_weight = ( rag[n1][n2]['weight'] - min_weight ) / ( max_weight - min_weight )
            out[line] = norm_weight*high_color + (1 - norm_weight)*low_color
        else:
            out[line] = low_color
            
        circle = draw.circle(r1,c1,2)
        out[circle] = node_color

    return out
