import networkx as nx
import _construct
from skimage import util


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

        # print "before ",self.order()
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


def rag_meancolor(img, labels):
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

    img = util.img_as_ubyte(img)
    if img.ndim == 4:
        return _construct.construct_rag_meancolor_3d(img, labels)
    elif img.ndim == 3:
        return _construct.construct_rag_meancolor_2d(img, labels)
    else:
        raise ValueError("Image dimension not supported")
