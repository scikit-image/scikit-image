import networkx as nx
import numpy as np


def threshold_cut(label_image, rag, thresh):
    """Combine regions seperated by weight less than threshold.

    Given an image's labels and its RAG, outputs new labels by
    combining regions whose nodes are seperated by a weight less
    than the given threshold.

    Parameters
    ----------
    label_image : (width, height) or (width, height, 3) ndarray
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

    map_array = np.arange(label_image.max() + 1, dtype=np.int)
    for i, nodes in enumerate(comps):
        for node in nodes:
            for label in rag.node[node]['labels']:
                map_array[label] = i

    return map_array[label_image]
