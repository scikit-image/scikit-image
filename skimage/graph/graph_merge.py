import numpy as np
import heapq


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


def _revalidate_node_edges(rag, node, heap_list):
    """Handles validation and invalidation of edges incident to a node.

    This function invalidates all existing edges incident on `node` and inserts
    new items in `heap_list` updated with the valid weights.

    rag : RAG
        The Region Adjacency Graph.
    node : int
        The id of the node whose incident edges are to be validated/invalidated
        .
    heap_list : list
        The list containing the existing heap of edges.
    """
    # networkx updates data dictionary if edge exists
    # this would mean we have to reposition these edges in
    # heap if their weight is updated.
    # instead we invalidate them

    for n in rag.neighbors(node):
        data = rag[node][n]
        try:
            # invalidate existing neghbors of `dst`, they have new weights
            data['heap item'][3] = False
        except KeyError:
            # will hangle the case where the edge did not exist in the existing
            # graph
            pass

        wt = data['weight']
        heap_item = [wt, node, n, True]
        data['heap item'] = heap_item
        heapq.heappush(heap_list, heap_item)


def merge_hierarchical_mean_color(labels, rag, thresh, in_place=True, merge_in_place=False):
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
    in_place : bool, optional
        If set, the RAG is modified in place.

    Examples
    --------
    >>> from skimage import data, graph, segmentation
    >>> img = data.coffee()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_mean_color(img, labels)
    >>> new_labels = graph.merge_hierarchical_mean_color(labels, rag, 40)
    """
    return merge_hierarchical(labels, rag, thresh, in_place, merge_in_place,
                              _pre_merge_mean_color, _weight_mean_color)


def merge_hierarchical(labels, rag, thresh, in_place, merge_in_place,pre_merge_func,
                       weight_func):
    """Perform hierarchical merging of a RAG.

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
    in_place : bool, optional
        If set, the RAG is modified in place.
    pre_merge_func : callable
        This function is called before merging two nodes. For the RAG `graph`
        while merging `src` and `dst`, it is called as follows
        ``pre_merge_func(graph, src, dst)``.
    weight_func : callable
        The function to compute the new weights of the nodes adjacent to the
        merged node. This is directly supplied as the argument `weight_func`
        to `merge_nodes`.

    Returns
    -------
    out : ndarray
        The new labeled array.

    """
    if not in_place:
        rag = rag.copy()

    edge_heap = []
    for n1, n2, data in rag.edges_iter(data=True):
        # Push a valid edge in the heap
        wt = data['weight']
        heap_item = [wt, n1, n2, data]
        heapq.heappush(edge_heap, heap_item)

        # Reference to the heap item in the graph
        data['heap item'] = heap_item

    while edge_heap[0][0] < thresh:
        _, n1, n2, valid = heapq.heappop(edge_heap)

        # Ensure popped edge is valid, if not, the edge is discarded
        if valid:
            pre_merge_func(rag, n1, n2)
            # Invalidate all neigbors of `src` before its deleted
            for n in rag.neighbors(n1):
                rag[n1][n]['heap item'][3] = False

            if not merge_in_place:
                for n in rag.neighbors(n2):
                    rag[n2][n]['heap item'][3] = False

            if not merge_in_place:
            
                #print 'added',next_id
                next_id = rag.next_id()

                rag._add_node(next_id)
                rag.node[next_id] = rag.node[n2]

                for nbr in rag.neighbors(n2):
                    rag.add_edge(nbr, next_id, {'weight':rag[n][n2]['weight']})
                
                rag.remove_node(n2)
                
                src, dst = n1, next_id
            else:
                src, dst = n1, n2
            
            new_id = rag.merge_nodes(src, dst, weight_func)
            
            _revalidate_node_edges(rag, new_id, edge_heap)

    arr = np.arange(labels.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes_iter(data=True)):
        for label in d['labels']:
            arr[label] = ix

    return arr[labels]
