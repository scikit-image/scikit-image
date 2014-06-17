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
        """Merges nodes `i` and `j`.

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

        if not self.has_edge(i, j):
            raise ValueError('Cant merge non adjacent nodes')

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


def _add_edge(values, g):
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
    `arr`

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

    """
    g = RAG()

    fp = np.zeros((3,) * arr.ndim)
    slc = slice(1, None, None)
    fp[(slc,) * arr.ndim] = 1

    filters.generic_filter(
        arr,
        function=_add_edge,
        footprint=fp,
        mode='constant',
        cval=-1,
        extra_arguments=(g,
                         ))
    iter = np.nditer(arr, flags=['multi_index'])

    while not iter.finished:

        current = arr[iter.multi_index]
        try:
            g.node[current]['pixel count'] += 1
            g.node[current]['total color'] += img[iter.multi_index]
        except KeyError:
            g.add_node(current)
            g.node[current]['pixel count'] = 1
            g.node[current]['total color'] = img[
                iter.multi_index].astype(np.long)
            g.node[current]['labels'] = [arr[iter.multi_index]]

        iter.iternext()

    for n in g.nodes():
        g.node[n]['mean color'] = g.node[n][
            'total color'] / g.node[n]['pixel count']

    for x, y in g.edges_iter():
        diff = g.node[x]['mean color'] - g.node[y]['mean color']
        g[x][y]['weight'] = np.linalg.norm(diff)

    return g
