try:
    import networkx as nx
except ImportError:
    from ..._shared.utils import warn
    warn('RAGs require networkx')
import numpy as np
from scipy import sparse
from . import _ncut_cy


def DW_matrices(graph):
    """Return the diagonal and weight matrices of a graph.

    Parameters
    ----------
    graph : RAG
        A Region Adjacency Graph.

    Returns
    -------
    D : csc_matrix
        The diagonal matrix of the graph. ``D[i, i]`` is the sum of weights of
        all edges incident on `i`. All other entries are `0`.
    W : csc_matrix
        The weight matrix of the graph. ``W[i, j]`` is the weight of the edge
        joining `i` to `j`.
    """
    # sparse.eighsh is most efficient with CSC-formatted input
    W = nx.to_scipy_sparse_matrix(graph, format='csc')
    entries = W.sum(axis=0)
    D = sparse.dia_matrix((entries, 0), shape=W.shape).tocsc()

    return D, W
