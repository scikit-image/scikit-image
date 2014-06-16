import networkx as nx
import numpy as np


def threshold_cut(label, rag, thresh):
    """Combines regions seperated by weight less than threshold.

    Given an image's labels and its RAG, outputs new labels by
    combining regions whose nodes are seperated by a weight less
    than the given threshold.

    Parameters
    ----------
    label : (width, height, 3) or (width, height, depth, 3) ndarray
        The array of labels.
    rag : RAG
        The region adjacency graph.
    thresh : float
        The threshold, regions with edge weights less than this
        are combined.

    Returns
    -------
    out : (width, height, 3) or (width, height, depth, 3) ndarray
        The new labelled array.

    Examples
    --------
    >>> from skimage import data,graph,segmentation
    >>> img = data.lena()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_meancolor(img, labels)
    >>> new_labels = graph.threshold_cut(labels, rag, 10)
    """
    to_remove = [(x, y) for x, y, d in rag.edges_iter(data=True)
                 if d['weight'] >= thresh]
    rag.remove_edges_from(to_remove)

    comps = nx.connected_components(rag)
    out = np.copy(label)

    for i, nodes in enumerate(comps):
        for node in nodes:
            for l in rag.node[node]['labels']:
                out[label == l] = i

    return out
