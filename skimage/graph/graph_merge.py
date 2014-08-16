import numpy as np
import heapq


def _hmerge_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

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
        The absolute difference of the mean color between node `dst` and `n`.
    """
    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return diff


def merge_hierarchical(labels, rag, thresh, in_place=True):
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
    in_place : bool, optional
        If set, the RAG is modified in place.

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

    if not in_place:
        rag = rag.copy()

    edge_heap = []
    for x, y, data in rag.edges_iter(data=True):
        if x != y:
            # Validate all edges and push them in heap
            data['valid'] = True
            wt = data['weight']
            heapq.heappush(edge_heap, (wt, x, y, data))

    while min_wt < thresh:
        min_wt, x, y, data = heapq.heappop(edge_heap)

        # Ensure popped edge is valid, if not, the edge is discarded
        if min_wt < thresh and data['valid']:
            total_color = (rag.node[y]['total color'] +
                           rag.node[x]['total color'])
            n_pixels = rag.node[x]['pixel count'] + rag.node[y]['pixel count']
            rag.node[y]['total color'] = total_color
            rag.node[y]['pixel count'] = n_pixels
            rag.node[y]['mean color'] = total_color / n_pixels

            # This will invalidate all the below edges in the heap
            for n in rag.neighbors(x):
                rag[x][n]['valid'] = False

            for n in rag.neighbors(y):
                rag[y][n]['valid'] = False

            rag.merge_nodes(x, y, _hmerge_mean_color)
            for n in rag.neighbors(y):
                if n != y:
                    # networkx updates data dictionary if edge exists
                    # this would mean we have to reposition these edges in
                    # heap if their weight is updated.
                    # instead we invalidate them

                    # invalidates the edge in the heap, if it all it exists
                    data = rag[y][n]
                    data['valid'] = False

                    # allocate a new dictionary for the edge
                    data_copy = data.copy()
                    rag[y][n] = data_copy
                    rag[n][y] = data_copy

                    # validate this edge
                    rag.add_edge(y, n, valid=True)

                    # push the new validated edge in the heap, this will be
                    # moved to its proper position
                    wt = rag[y][n]['weight']
                    heapq.heappush(edge_heap, (wt, y, n, rag[y][n]))

    arr = np.arange(labels.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes_iter(data=True)):
        for label in d['labels']:
            arr[label] = ix

    return arr[labels]
