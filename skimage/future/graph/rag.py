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
from ... import draw, measure, segmentation, util, color
try:
    from matplotlib import colors
    from matplotlib import cm
except ImportError:
    pass


def min_weight(graph, src, dst, n):
    """Callback to handle merging nodes by choosing minimum weight.

    Returns either the weight between (`src`, `n`) or (`dst`, `n`)
    in `graph` or the minimum of the two when both exist.

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
        minimum of the two when both exist.

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

    def __init__(self, data=None, **attr):

        super(RAG, self).__init__(data, **attr)
        try:
            self.max_id = max(self.nodes_iter())
        except ValueError:
            # Empty sequence
            self.max_id = 0

    def merge_nodes(self, src, dst, weight_func=min_weight, in_place=True,
                    extra_arguments=[], extra_keywords={}):
        """Merge node `src` and `dst`.

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
        in_place : bool, optional
            If set to `True`, the merged node has the id `dst`, else merged
            node has a new id which is returned.
        extra_arguments : sequence, optional
            The sequence of extra positional arguments passed to
            `weight_func`.
        extra_keywords : dictionary, optional
            The dict of keyword arguments passed to the `weight_func`.

        Returns
        -------
        id : int
            The id of the new node.

        Notes
        -----
        If `in_place` is `False` the resulting node has a new id, rather than
        `dst`.
        """
        src_nbrs = set(self.neighbors(src))
        dst_nbrs = set(self.neighbors(dst))
        neighbors = (src_nbrs | dst_nbrs) - set([src, dst])

        if in_place:
            new = dst
        else:
            new = self.next_id()
            self.add_node(new)

        for neighbor in neighbors:
            w = weight_func(self, src, new, neighbor, *extra_arguments,
                            **extra_keywords)
            self.add_edge(neighbor, new, weight=w)

        self.node[new]['labels'] = (self.node[src]['labels'] +
                                    self.node[dst]['labels'])
        self.remove_node(src)

        if not in_place:
            self.remove_node(dst)

        return new

    def add_node(self, n, attr_dict=None, **attr):
        """Add node `n` while updating the maximum node id.

        .. seealso:: :func:`networkx.Graph.add_node`."""
        super(RAG, self).add_node(n, attr_dict, **attr)
        self.max_id = max(n, self.max_id)

    def add_edge(self, u, v, attr_dict=None, **attr):
        """Add an edge between `u` and `v` while updating max node id.

        .. seealso:: :func:`networkx.Graph.add_edge`."""
        super(RAG, self).add_edge(u, v, attr_dict, **attr)
        self.max_id = max(u, v, self.max_id)

    def copy(self):
        """Copy the graph with its max node id.

        .. seealso:: :func:`networkx.Graph.copy`."""
        g = super(RAG, self).copy()
        g.max_id = self.max_id
        return g

    def next_id(self):
        """Returns the `id` for the new node to be inserted.

        The current implementation returns one more than the maximum `id`.

        Returns
        -------
        id : int
            The `id` of the new node to be inserted.
        """
        return self.max_id + 1

    def _add_node_silent(self, n):
        """Add node `n` without updating the maximum node id.

        This is a convenience method used internally.

        .. seealso:: :func:`networkx.Graph.add_node`."""
        super(RAG, self).add_node(n)


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
        if value != current:
            graph.add_edge(current, value)

    return 0


def rag_mean_color(image, labels, connectivity=2, mode='distance',
                   sigma=255.0):
    """Compute the Region Adjacency Graph using mean colors.

    Given an image and its initial segmentation, this method constructs the
    corresponding Region Adjacency Graph (RAG). Each node in the RAG
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
    >>> from skimage import data, segmentation
    >>> from skimage.future import graph
    >>> img = data.astronaut()
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


def draw_rag(labels, rag, img, border_color=None, node_color='#ffff00',
             edge_color='#00ff00', colormap=None, thresh=np.inf,
             desaturate=False, in_place=True):
    """Draw a Region Adjacency Graph on an image.

    Given a labelled image and its corresponding RAG, draw the nodes and edges
    of the RAG on the image with the specified colors. Nodes are marked by
    the centroids of the corresponding regions.

    Parameters
    ----------
    labels : ndarray, shape (M, N)
        The labelled image.
    rag : RAG
        The Region Adjacency Graph.
    img : ndarray, shape (M, N, 3)
        Input image.
    border_color : colorspec, optional
        Any matplotlib colorspec.
    node_color : colorspec, optional
        Any matplotlib colorspec. Yellow by default.
    edge_color : colorspec, optional
        Any matplotlib colorspec. Green by default.
    colormap : colormap, optional
        Any matplotlib colormap. If specified the edges are colormapped with
        the specified color map.
    thresh : float, optional
        Edges with weight below `thresh` are not drawn, or considered for color
        mapping.
    desaturate : bool, optional
        Convert the image to grayscale before displaying. Particularly helps
        visualization when using the `colormap` option.
    in_place : bool, optional
        If set, the RAG is modified in place. For each node `n` the function
        will set a new attribute ``rag.node[n]['centroid']``.

    Returns
    -------
    out : ndarray, shape (M, N, 3)
        The image with the RAG drawn.

    Examples
    --------
    >>> from skimage import data, segmentation
    >>> from skimage.future import graph
    >>> img = data.coffee()
    >>> labels = segmentation.slic(img)
    >>> g =  graph.rag_mean_color(img, labels)
    >>> out = graph.draw_rag(labels, g, img)
    """
    if not in_place:
        rag = rag.copy()

    if desaturate:
        img = color.rgb2gray(img)
        img = color.gray2rgb(img)

    out = util.img_as_float(img, force_copy=True)
    cc = colors.ColorConverter()

    edge_color = cc.to_rgb(edge_color)
    node_color = cc.to_rgb(node_color)

    # Handling the case where one node has multiple labels
    # offset is 1 so that regionprops does not ignore 0
    offset = 1
    map_array = np.arange(labels.max() + 1)
    for n, d in rag.nodes_iter(data=True):
        for label in d['labels']:
            map_array[label] = offset
        offset += 1

    rag_labels = map_array[labels]
    regions = measure.regionprops(rag_labels)

    for (n, data), region in zip(rag.nodes_iter(data=True), regions):
        data['centroid'] = region['centroid']

    if border_color is not None:
        border_color = cc.to_rgb(border_color)
        out = segmentation.mark_boundaries(out, rag_labels, color=border_color)

    if colormap is not None:
        edge_weight_list = [d['weight'] for x, y, d in
                            rag.edges_iter(data=True) if d['weight'] < thresh]
        norm = colors.Normalize()
        norm.autoscale(edge_weight_list)
        smap = cm.ScalarMappable(norm, colormap)

    for n1, n2, data in rag.edges_iter(data=True):

        if data['weight'] >= thresh:
            continue
        r1, c1 = map(int, rag.node[n1]['centroid'])
        r2, c2 = map(int, rag.node[n2]['centroid'])
        line = draw.line(r1, c1, r2, c2)

        if colormap is not None:
            out[line] = smap.to_rgba([data['weight']])[0][:-1]
        else:
            out[line] = edge_color

        circle = draw.circle(r1, c1, 2)
        out[circle] = node_color

    return out
