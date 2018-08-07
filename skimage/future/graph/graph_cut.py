try:
    import networkx as nx
except ImportError:
    from ..._shared.utils import warn
    warn('RAGs require networkx')
import numpy as np
from . import _ncut
from . import _ncut_cy
from scipy.sparse.linalg import eigsh


def cut_threshold(labels, rag, thresh, in_place=True):
    """Combine regions separated by weight less than threshold.

    Given an image's labels and its RAG, output new labels by
    combining regions whose nodes are separated by a weight less
    than the given threshold.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The region adjacency graph.
    thresh : float
        The threshold. Regions connected by edges with smaller weights are
        combined.
    in_place : bool
        If set, modifies `rag` in place. The function will remove the edges
        with weights less that `thresh`. If set to `False` the function
        makes a copy of `rag` before proceeding.

    Returns
    -------
    out : ndarray
        The new labelled array.

    Examples
    --------
    >>> from skimage import data, segmentation
    >>> from skimage.future import graph
    >>> img = data.astronaut()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_mean_color(img, labels)
    >>> new_labels = graph.cut_threshold(labels, rag, 10)

    References
    ----------
    .. [1] Alain Tremeau and Philippe Colantoni
           "Regions Adjacency Graph Applied To Color Image Segmentation"
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.11.5274
    """
    if not in_place:
        rag = rag.copy()

    # Because deleting edges while iterating through them produces an error.
    to_remove = [(x, y) for x, y, d in rag.edges(data=True)
                 if d['weight'] >= thresh]
    rag.remove_edges_from(to_remove)

    comps = nx.connected_components(rag)

    # We construct an array which can map old labels to the new ones.
    # All the labels within a connected component are assigned to a single
    # label in the output.
    map_array = np.arange(labels.max() + 1, dtype=labels.dtype)
    for i, nodes in enumerate(comps):
        for node in nodes:
            for label in rag.node[node]['labels']:
                map_array[label] = i

    return map_array[labels]


def cut_normalized(labels, rag, thresh=0.001, in_place=True,
                   max_edge=1.0):
    """Perform Normalized Graph cut on the Region Adjacency Graph.

    Given an image's labels and its similarity RAG, recursively perform
    a two-way normalized cut on it. All nodes belonging to a subgraph
    that cannot be cut further are assigned a unique label in the
    output.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The region adjacency graph.
    thresh : float
        The threshold. A subgraph won't be further subdivided if the
        value of the N-cut exceeds `thresh`.
    in_place : bool
        If set, modifies `rag` in place. For each node `n` the function will
        set a new attribute ``rag.node[n]['ncut label']``.
    max_edge : float, optional
        The maximum possible value of an edge in the RAG. This corresponds to
        an edge between identical regions. This is used to put self
        edges in the RAG.

    Returns
    -------
    out : ndarray
        The new labeled array.

    Examples
    --------
    >>> from skimage import data, segmentation
    >>> from skimage.future import graph
    >>> img = data.astronaut()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_mean_color(img, labels, mode='similarity')
    >>> new_labels = graph.cut_normalized(labels, rag)

    References
    ----------
    .. [1] Shi, J.; Malik, J., "Normalized cuts and image segmentation",
           Pattern Analysis and Machine Intelligence,
           IEEE Transactions on, vol. 22, no. 8, pp. 888-905, August 2000.

    """
    gen = cut_normalized_gen(labels, rag, thresh, in_place, max_edge)
    labels = next(gen)
    return labels


def cut_normalized_gen(labels, rag, init_thresh=0.001,
                       in_place=True, max_edge=1.0):
    """Generate normalized graph cuts on the region adjacency graph.

    Given an image's labels and its similarity RAG, recursively perform
    a two-way normalized cut on it. All nodes belonging to a subgraph
    that cannot be cut further are assigned a unique label in the
    output. Appending a new value to the thresh list and calling
    next on this generator returns a new array of labels updated
    for the new thresh value.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The region adjacency graph.
    init_thresh : float
        Initial threshold value to segment with. A subgraph won't be further
        subdivided if the value of the N-cut exceeds a threshold value.
    in_place : bool
        If set, modifies `rag` in place. For each node `n` the function will
        set a new attribute ``rag.node[n]['ncut label']``.
    max_edge : float, optional
        The maximum possible value of an edge in the RAG. This corresponds to
        an edge between identical regions. This is used to put self
        edges in the RAG.

    Yields
    ------
    out : ndarray
        The new labeled array.

    Examples
    --------
    >>> from skimage import data, segmentation
    >>> from skimage.future import graph
    >>> img = data.astronaut()
    >>> labels = segmentation.slic(img, compactness=30, n_segments=400)
    >>> rag = graph.rag_mean_color(img, labels, mode='similarity')
    >>>
    >>> init_thresh = 1e-5
    >>> new_label_gen = graph.cut_normalized_gen(labels, rag, init_thresh)
    >>> new_labels = next(new_label_gen)
    >>> new_labels = new_label_gen.send(1e-3)
    >>> new_labels = new_label_gen.send(1e-4)

    References
    ----------
    .. [1] Shi, J.; Malik, J., "Normalized cuts and image segmentation",
           Pattern Analysis and Machine Intelligence,
           IEEE Transactions on, vol. 22, no. 8, pp. 888-905, August 2000.
    """
    if not in_place:
        rag = rag.copy()

    for node in rag.nodes():
        rag.add_edge(node, node, weight=max_edge)

    # Initialize the generator
    ncut_gen = _ncut_relabel(rag, 0)
    next(ncut_gen)

    # Generate new labels as thresholds come
    thresh = init_thresh
    while True:
        if thresh is None:
            raise ValueError('Once generator initialized by calling next(gen),'
                             ' use gen.send(thresh) to modify threshold.')
        ncut_gen.send(thresh)

        map_array = np.zeros(labels.max() + 1, dtype=labels.dtype)
        # Mapping from old labels to new
        for n, d in rag.nodes(data=True):
            map_array[d['labels']] = d['ncut label']

        thresh = yield map_array[labels]


def partition_by_cut(cut, rag):
    """Compute resulting subgraphs from given bi-partition.

    Parameters
    ----------
    cut : array
        A array of booleans. Elements set to `True` belong to one
        set.
    rag : RAG
        The Region Adjacency Graph.

    Returns
    -------
    sub1, sub2 : RAG
        The two resulting subgraphs from the bi-partition.
    """
    # `cut` is derived from `D` and `W` matrices, which also follow the
    # ordering returned by `rag.nodes()` because we use
    # nx.to_scipy_sparse_matrix.

    # Example
    # rag.nodes() = [3, 7, 9, 13]
    # cut = [True, False, True, False]
    # nodes1 = [3, 9]
    # nodes2 = [7, 10]

    nodes1 = [n for i, n in enumerate(rag.nodes()) if cut[i]]
    nodes2 = [n for i, n in enumerate(rag.nodes()) if not cut[i]]

    sub1 = rag.subgraph(nodes1)
    sub2 = rag.subgraph(nodes2)

    return sub1, sub2


def _get_partition_vector(d, w):
    """
    Get vector used to partition a graph into two subgraphs according
    to the normalized cuts algorithm.

    Parameters
    ----------
    d : ndarray
        The diagonal matrix of the graph.
    w : ndarray
        The weight matrix of the graph.

    Returns
    -------
    part_vec : array
        The vector to threshold.
    """
    m = w.shape[0]
    d2 = d.copy()

    # Since d is diagonal, we can directly operate on its data
    # the inverse of the square root
    d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)

    # Refer Shi & Malik 2000, Equation 7, Page 891
    v0 = 2 * np.random.random((m,)) - 1
    v0 /= np.linalg.norm(v0)
    vals, vectors = eigsh(d2 * (d - w) * d2, which='SM',
                          k=min(m-1, 100), v0=v0)

    # Build vector orthogonal to d from first two eigenvectors
    # Helps avoid problems from degeneracy
    # Refer Shi & Malik 2000, Section 2.1.10, Page 891
    # Refer Shi & Malik 2000, Section 3.2.3, Page 893
    vectors = (vectors[:, :2].T / d2.data).T
    data_normed = d.data / np.dot(d.data, d.data)
    part_vec = ((vectors[:, 0] - np.dot(vectors[:, 0], d.data) * data_normed) +
                (vectors[:, 1] - np.dot(vectors[:, 1], d.data) * data_normed))
    return part_vec


def get_min_ncut(part_vec, w):
    """
    Find the minimum ncut and the partition it by effectively thresholding
    the values in the supplied partition vector

    Parameters
    ----------
    part_vec : array
        This vector's values thresholded to decide partition
    w : ndarray
        The weight matrix of the graph.

    Returns
    -------
    min_mask : array
        The array of booleans which denotes the bi-partition.
    mcut : float
        The value of the minimum ncut.
    """
    min_mask = np.zeros_like(part_vec, dtype=np.bool)

    # Get sorted indices and make map from unsorted
    # to sorted indices
    ind_sorted = np.argsort(part_vec)
    ind_sort = np.empty(len(ind_sorted), dtype=np.int32)
    ind_sort[ind_sorted] = np.array(range(len(ind_sorted)))

    # Refer Shi & Malik 2000, Section 3.1.3, Page 892
    # Find the optimal partitioning
    mcut, mcol = _ncut_cy.find_cut(ind_sorted, ind_sort, w)
    min_mask[ind_sorted[:mcol + 1]] = True

    return min_mask, mcut


def _label_all(rag, attr_name):
    """Assign a unique integer to the given attribute in the RAG.

    This function assumes that all labels in `rag` are unique. It
    picks up a random label from them and assigns it to the `attr_name`
    attribute of all the nodes.

    rag : RAG
        The Region Adjacency Graph.
    attr_name : string
        The attribute to which a unique integer is assigned.
    """
    node = min(rag.nodes())
    new_label = rag.node[node]['labels'][0]
    for n, d in rag.nodes(data=True):
        d[attr_name] = new_label


def _ncut_relabel(rag, init_thresh):
    """Perform Normalized Graph cut on the Region Adjacency Graph.

    Recursively partition the graph into two, until further subdivision
    yields a cut greater than `thresh` or such a cut cannot be computed.
    For such a subgraph, indices to labels of all its nodes map to a single
    unique value.

    Parameters
    ----------
    rag : RAG
        The region adjacency graph.
    init_thresh : float
        The initial threshold. A subgraph won't be further subdivided if the
        value of the N-cut exceeds `thresh`.
    """
    thresh = init_thresh
    d, w = _ncut.DW_matrices(rag)

    # If 2 regions not cutting is optimal choice
    if w.shape[0] > 2:
        part_vec = _get_partition_vector(d, w)
        cut_mask, mcut = get_min_ncut(part_vec, w)

        calc_subgraph = True
        while True:
            # mcut exceeded thresh. Update the labels in the subgraph
            # and wait until thresh rises above mcut
            if (mcut >= thresh):
                _label_all(rag, 'ncut label')
            while (mcut >= thresh):
                thresh = yield

            # Only prepare _ncut_relabel generators for subgraphs
            # once, and only if necessary
            if calc_subgraph:
                sub1, sub2 = partition_by_cut(cut_mask, rag)
                branch1 = _ncut_relabel(sub1, thresh)
                branch2 = _ncut_relabel(sub2, thresh)
                next(branch1)
                next(branch2)
                del cut_mask
                calc_subgraph = False

            # Propagate send() call to subgraphs
            while (mcut < thresh):
                branch1.send(thresh)
                branch2.send(thresh)
                thresh = yield
    else:
        # There were less than three super pixels to cut. The remaining
        # graph is a region.
        # Assign `ncut label` by picking any label from the existing
        # nodes, since `labels` are unique, `new_label` is also unique.
        while True:
            _label_all(rag, 'ncut label')
            yield
