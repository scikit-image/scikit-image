import networkx as nx
import numpy as np
from scipy.ndimage import filters
from scipy import ndimage as nd


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


def rag_mean_color(image, labels, connectivity=2):
    """Compute the Region Adjacency Graph using mean colors.

    Given an image and its initial segmentation, this method constructs the
    corresponsing Region Adjacency Graph (RAG). Each node in the RAG
    represents a set of pixels within `image` with the same label in `labels`.
    The weight between two adjacent regions is the difference in their mean
    color.

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

    for x, y in graph.edges_iter():
        diff = graph.node[x]['mean color'] - graph.node[y]['mean color']
        graph[x][y]['weight'] = np.linalg.norm(diff)

    return graph
