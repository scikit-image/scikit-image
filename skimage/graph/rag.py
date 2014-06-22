import networkx as nx
import numpy as np
from scipy.ndimage import filters


class RAG(nx.Graph):

    """
    The class for holding the Region Adjacency Graph (RAG).

    Each region is a contiguous set of pixels in an image, usuall
    sharing some common property.Adjacent regions have an edge
    between their corresponding nodes.
    """

    def merge_nodes(self, i, j):
        """Merge node `i` into `j`.

        The new combined node is adjacent to all the neighbors of `i`
        and `j`. In case of conflicting edges, edge with higher weight
        is chosen.

        Parameters
        ----------
        i : int
            Node to be merged.
        j : int
            Node to be merged.

        """
        for x in self.neighbors(i):
            if x == j:
                continue
            w1 = self.get_edge_data(x, i)['weight']
            w2 = -1
            if self.has_edge(x, j):
                w2 = self.get_edge_data(x, j)['weight']

            w = max(w1, w2)
            self.add_edge(x, j, weight=w)

        self.node[j]['labels'] += self.node[i]['labels']
        self.remove_node(i)


def _add_edge_filter(values, g):
    """Adds an edge between first element in `values` and
    all other elements of `values` in the graph `g`.

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
        if value >= 0:
            g.add_edge(current, value)

    return 0.0


def rag_meancolor(img, arr):
    """Computes the Region Adjacency Graph of a color image using
    difference in mean color of regions as edge weights.

    Given an image and its segmentation, this method constructs the
    corresponsing Region Adjacency Graph (RAG).Each node in the RAG
    represents a contiguous pixels with in `img` the same label in
    `arr`.

    Parameters
    ----------
    img : (width, height, 3) or (width, height, depth, 3) ndarray
        Input image.
    arr : (width, height) or (width, height, depth) ndarray
        The array with labels.

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

    fp = np.zeros((3,) * arr.ndim)
    slc = slice(1, None, None)
    fp[(slc,) * arr.ndim] = 1

    # The footprint is constructed in such a way that the first
    # element in the array being passed to _add_edge_filter is
    # the central value.
    filters.generic_filter(
        arr,
        function=_add_edge_filter,
        footprint=fp,
        mode='constant',
        cval=-1,
        extra_arguments=(g,))

    for index in np.ndindex(arr.shape):
        current = arr[index]

        if 'pixel count' in g.node[current]:
            g.node[current]['pixel count'] += 1
            g.node[current]['total color'] += img[index]
        else:
            g.node[current]['pixel count'] = 1
            g.node[current]['total color'] = img[index].astype(np.double)
            g.node[current]['labels'] = [arr[index]]

    for n in g:
        g.node[n]['mean color'] = (g.node[n]['total color'] /
                                   g.node[n]['pixel count'])

    for x, y in g.edges_iter():
        diff = g.node[x]['mean color'] - g.node[y]['mean color']
        g[x][y]['weight'] = np.linalg.norm(diff)

    return g
