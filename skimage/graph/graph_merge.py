import numpy as np


def _hmerge(rag, x, y, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `y` is already computed.

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
        The absolute difference of the mean color between node `y` and `n`.
    """
    diff = rag.node[y]['mean color'] - rag.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return diff


def merge_hierarchical(labels, rag, thresh):
    """Perform hierarchical merging of a RAG.

    Given an image's labels and its RAG, the method merges the similar nodes
    until the weight between every two nodes is more than `thresh`.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The Region Adjacency Graph.
    thresh : float
        The threshold. Regions connected by an edges with smaller wegiht than
        `thresh` are merged. A high value of `thresh` would mean that a lot of
        regions are merged, and the output will contain fewer regions.

    Returns
    -------
    out : ndarray
        The new labelled array.

    Examples
    --------
    >>> from skimage import data, graph, segmentation
    >>> img = data.coffee()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_mean_color(img, labels)
    >>> new_labels = graph.merge_hierarchical(labels, rag, 40)
    """
    min_wt = 0
    while min_wt < thresh:

        valid_edges = ((x, y, d) for x, y, d in rag.edges(data=True) if x != y)
        x, y, d = min(valid_edges, key=lambda x: x[2]['weight'])
        min_wt = d['weight']

        if min_wt < thresh:
            total_color = (rag.node[y]['total color'] +
                           rag.node[x]['total color'])
            n_pixels = rag.node[x]['pixel count'] + rag.node[y]['pixel count']
            rag.node[y]['total color'] = total_color
            rag.node[y]['pixel count'] = n_pixels
            rag.node[y]['mean color'] = total_color / n_pixels

            rag.merge_nodes(x, y, _hmerge)

    arr = np.arange(labels.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes_iter(data=True)):
        for label in d['labels']:
            arr[label] = ix

    return arr[labels]
