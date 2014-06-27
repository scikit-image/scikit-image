import networkx as nx
import numpy as np
from scipy.ndimage import filters
from scipy import ndimage as nd


def min_weight(g, src, dst, n):
    """Callback to handle merging nodes by choosing minimum weight.

    Returns either the weight between (`src`, `n`) or (`dst`, `n`)
    in `g` or the minumum of the two when both exist.

    Parameters
    ----------
    g : RAG
        The graph to consider.
    src, dst : int
        Typically the verices in `g` to be merged.
    n : int
        A neighbor of `src` or `dst` or both

    Returns
    -------
    weight : float
        The weight between (`src`, `n`) or (`dst`, `n`) in `g` or the
        minumum of the two when both exist.

    """

    # cover the cases where n only has edge to either `src` or `dst`
    w1 = g[n].get(src, {'weight': np.inf})['weight']
    w2 = g[n].get(dst, {'weight': np.inf})['weight']
    return min(w1, w2)


class RAG(nx.Graph):

    """
    The Region Adjacency Graph (RAG) of an image.
    """

    def merge_nodes(self, src, dst, weight_func=min_weight, extra_arguments=[],
                    extra_keywords={}):
        """Merge two nodes.

        The new combined node is adjacent to all the neighbors of `src`
        and `dst`. `weight_func` is called to decide the weight of edges
        incident on the new node.

        Parameters
        ----------
        src, dst : int
            Nodes to be merged. The resulting node will have ID `dst`.
        weight_func : callable, optional
            Function to decide edge weight of edges incident on the new node.
            For each neighbor `n` for `src and `dst`, `weight_func` will be
            called as follows: `weight_func(src, dst, n, *extra_arguments,
            **extra_keywords)`
        extra_arguments : sequence, optional
            The sequence of extra positional arguments passed to
            `weight_func`.
        extra_keywords :
            The dict of keyword arguments passed to the `weight_func`.

        """
        neighbors = self.adj[src].copy()
        neighbors.update(self.adj[dst])

        try:
            del neighbors[src]
        except KeyError:
            pass
        try:
            del neighbors[dst]
        except KeyError:
            pass

        for neighbor in neighbors:
            w = weight_func(self, src, dst, neighbor, *extra_arguments,
                            **extra_keywords)
            self.add_edge(neighbor, dst, weight=w)

        self.node[dst]['labels'] += self.node[src]['labels']
        self.remove_node(src)


def _add_edge_filter(values, g):
    """Create edge in `g` between first element of `values` and the rest.

    Add an edge between first element in `values` and
    all other elements of `values` in the graph `g`. `values[0]`
    is expected to be the central value of the footprint used.

    Parameters
    ----------
    values : array
        The array to process.
    g : RAG
        The graph to add edges in.

    Returns
    -------
    0 : int
        Always returns 0.

    """
    values = values.astype(int)
    current = values[0]
    for value in values[1:]:
        g.add_edge(current, value)

    return 0


def rag_meancolor(image, labels, connectivity=2):
    """Compute the Region Adjacency Graph using mean colors.

    Given an image and its initial segmentation, this method constructs the
    corresponsing Region Adjacency Graph (RAG). Each node in the RAG
    represents a set pixels within `image` with the same
    label in `labels`. The weight between two adjacent regions is the
    difference in their mean color.

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
    >>> rag = graph.rag_meancolor(img, labels)

    References
    ----------
    .. [1] Alain Tremeau and Philippe Colantoni
           "Regions Adjacency Graph Applied To Color Image Segmentation"
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.11.5274

    """
    g = RAG()

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
        extra_arguments=(g,))

    for n in g:
        g.node[n].update({'labels': [n],
                          'pixel count': 0,
                          'total color': np.array([0, 0, 0], dtype=np.double)})

    for index in np.ndindex(labels.shape):
        current = labels[index]
        g.node[current]['pixel count'] += 1
        g.node[current]['total color'] += image[index]

    for n in g:
        g.node[n]['mean color'] = (g.node[n]['total color'] /
                                   g.node[n]['pixel count'])

    for x, y in g.edges_iter():
        diff = g.node[x]['mean color'] - g.node[y]['mean color']
        g[x][y]['weight'] = np.linalg.norm(diff)

    return g
