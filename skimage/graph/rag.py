import networkx as nx
import numpy as np
from scipy.ndimage import filters
from scipy import ndimage as nd


class RAG(nx.Graph):

    """
    The class for holding the Region Adjacency Graph (RAG).

    Each region is a contiguous set of pixels in an image, usually
    sharing some common property. Adjacent regions have an edge
    between their corresponding nodes.
    """

    def merge_nodes(self, i, j, function=None, extra_arguments=[],
                    extra_keywords={}):
        """Merge node `i` into `j`.

        The new combined node is adjacent to all the neighbors of `i`
        and `j`. In case of conflicting edges the given function is
        called.

        Parameters
        ----------
        i, j : int
            Nodes to be merged. The resulting node will have ID `j`.
        function : callable, optional
            Function to decide which edge weight to keep when a node is
            adjacent to both `i` and `j`. The arguments passed to the
            function are, the tuples represnting both the conflicting edges
            and the graph.The default behaviour is that the edge with higher
            weight is kept.
        extra_arguments : sequence, optional
            The sequence of extra positional arguments passed to
            `function`
        extra_keywords :
            The dict of keyword arguments passed to the `function`.
        """
        for x in self.neighbors(i):
            if x == j:
                continue
            w1 = self.get_edge_data(x, i)['weight']
            w2 = -1
            if self.has_edge(x, j):
                w2 = self.get_edge_data(x, j)['weight']

            w = w1
            if w2 > 0:
                if not function:
                    w = max(w1, w2)
                else:
                    w = function((i, x), (j, x), self,
                                 *extra_arguments, **extra_keywords)
            self.add_edge(x, j, weight=w)

        self.node[j]['labels'] += self.node[i]['labels']
        self.remove_node(i)


def _add_edge_filter(values, g):
    """Add an edge between first element in `values` and
    all other elements of `values` in the graph `g`.`values[0]`
    is expected to be the central value of the footprint used.

    Parameters
    ----------
    values : array
        The array to process.
    g : RAG
        The graph to add edges in.

    Returns
    -------
    0.0 : float
        Always returns 0.

    """
    values = values.astype(int)
    current = values[0]
    for value in values[1:]:
        g.add_edge(current, value)

    return 0.0


def rag_meancolor(image, label_image, connectivity=2):
    """Compute the Region Adjacency Graph of a color image using
    difference in mean color of regions as edge weights.

    Given an image and its segmentation, this method constructs the
    corresponsing Region Adjacency Graph (RAG). Each node in the RAG
    represents a contiguous pixels with in `img` the same label in
    `arr`.

    Parameters
    ----------
    image : (width, height, 3) or (width, height, depth, 3) ndarray
        Input image.
    label_image : (width, height) or (width, height, depth) ndarray
        The array with labels.
    connectivity : float, optional
        Pixels with a squared distance less than `connectivity`from each other
        are considered adjacent.

    Returns
    -------
    out : RAG
        The region adjacency graph.

    Examples
    --------
    >>> from skimage import data,graph,segmentation
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

    fp = nd.generate_binary_structure(label_image.ndim, connectivity)
    for d in range(fp.ndim):
        fp = fp.swapaxes(0, d)
        fp[0, ...] = 0
        fp = fp.swapaxes(0, d)

    # The footprint is constructed in such a way that the first
    # element in the array being passed to _add_edge_filter is
    # the central value.

    for i in range(label_image.max() + 1):
        g.add_node(
            i, {'labels': [i], 'pixel count': 0, 'total color':
                np.array([0, 0, 0], dtype=np.double)})

    filters.generic_filter(
        label_image,
        function=_add_edge_filter,
        footprint=fp,
        mode='nearest',
        extra_arguments=(g,))

    for index in np.ndindex(label_image.shape):
        current = label_image[index]

        # if 'pixel count' in g.node[current]:
        g.node[current]['pixel count'] += 1
        g.node[current]['total color'] += image[index]
        # else:
        #    g.node[current]['pixel count'] = 1
        #    g.node[current]['total color'] = image[index].astype(np.double)
        #    g.node[current]['labels'] = [current]

    for n in g:
        g.node[n]['mean color'] = (g.node[n]['total color'] /
                                   g.node[n]['pixel count'])

    for x, y in g.edges_iter():
        diff = g.node[x]['mean color'] - g.node[y]['mean color']
        g[x][y]['weight'] = np.linalg.norm(diff)

    return g
