import networkx as nx
import numpy as np


def threshold_cut(labels, rag, thresh):
    """Combine regions seperated by weight less than threshold.

    Given an image's labels and its RAG, output new labels by
    combining regions whose nodes are seperated by a weight less
    than the given threshold.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The region adjacency graph.
    thresh : float
        The threshold, regions with edge weights less than this
        are combined.

    Returns
    -------
    out : ndarray
        The new labelled array.

    Examples
    --------
    >>> from skimage import data, graph, segmentation
    >>> img = data.lena()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_meancolor(img, labels)
    >>> new_labels = graph.threshold_cut(labels, rag, 10)

    References
    ----------
    .. [1] Alain Tremeau and Philippe Colantoni
           "Regions Adjacency Graph Applied To Color Image Segmentation"
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.11.5274

    """
    # Because deleting edges while iterating through them produces an error.
    to_remove = [(x, y) for x, y, d in rag.edges_iter(data=True)
                 if d['weight'] >= thresh]
    rag.remove_edges_from(to_remove)

    comps = nx.connected_components(rag)

    map_array = np.arange(labels.max() + 1, dtype=labels.dtype)
    for i, nodes in enumerate(comps):
        for node in nodes:
            for label in rag.node[node]['labels']:
                map_array[label] = i

    return map_array[labels]
