"""
===========
RAG Merging
===========

This example constructs a Region Adjacency Graph (RAG) and progressively merges
regions that are similar in color. Merging two adjacent regions produces
a new region with all the pixels from the merged regions. Regions are merged
until no highly similar region pairs remain.

"""

from skimage import graph, data, io, segmentation, color
import numpy as np


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    weight : float
        The absolute difference of the mean color between node `dst` and `n`.
    """

    #print 'merging
    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return diff


def _pre_merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])


def merge_hierarchical_mean_color(labels, rag, thresh, rag_copy=True,
                                  in_place_merge=False):
    """Perform hierarchical merging of a color distance RAG.

    Greedily merges the most similar pair of nodes until no edges lower than
    `thresh` remain.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The Region Adjacency Graph.
    thresh : float
        Regions connected by an edge with weight smaller than `thresh` are
        merged.
    rag_copy : bool, optional
        If set, the RAG copied before modifying.
    in_place_merge : bool, optional
        If set, the nodes are merged in place. Otherwise, a new node is
        created for each merge.

    Examples
    --------
    >>> from skimage import data, graph, segmentation
    >>> img = data.coffee()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_mean_color(img, labels)
    >>> new_labels = graph.merge_hierarchical_mean_color(labels, rag, 40)
    """
    return graph.merge_hierarchical(labels, rag, thresh, rag_copy,
                                    in_place_merge, _pre_merge_mean_color,
                                    _weight_mean_color)

img = data.coffee()
labels = segmentation.slic(img, compactness=30, n_segments=400)
g = graph.rag_mean_color(img, labels)
labels2 = merge_hierarchical_mean_color(labels, g, 40)
g2 = graph.rag_mean_color(img, labels2)

out = color.label2rgb(labels2, img, kind='avg')
out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
io.imshow(out)
io.show()
