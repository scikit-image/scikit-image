import networkx as nx
import numpy as np
from scipy import sparse
from . import _ncut_cy


def DW_matrices(graph):
    """Returns the diagonal and weight matrices of a graph.

    Parameters
    ----------
    graph : RAG
        A Region Adjacency Graph.

    Returns
    -------
    D : csc_matrix
        The diagonal matrix of the graph. `D[i, i]` is the sum of weights of
        all edges incident on `i`. All other enteries are `0`.
    W : csc_matrix
        The weight matrix of the graph. `W[i, j]` is the weight of the edge
        joining `i` to `j`.
    """
    #Cause sparse.eigsh prefers CSC format
    W = nx.to_scipy_sparse_matrix(graph, format='csc')
    entries = W.sum(axis=0)
    D = sparse.dia_matrix((entries, 0), shape=W.shape).tocsc()
    return D, W


def ncut_cost(mask, D, W):
    """Returns the N-cut cost of a bi-partition of a graph.

    Parameters
    ----------
    mask : ndarray
        The mask for the nodes in the graph. Nodes corrsesponding to a `True`
        value are in one set.
    D : csc_matrix
        The diagonal matrix of the graph.
    W : csc_matrix
        The weight matrix of the graph.

    Returns
    -------
    cost : float
        The cost of performing the N-cut.
    """
    mask = np.array(mask)
    cut = _ncut_cy.cut_cost(mask, W)

    # Cause D has elements only along diagonal
    assoc_a = D.data[mask].sum()
    assoc_b = D.data[np.logical_not(mask)].sum()

    return (cut / assoc_a) + (cut / assoc_b)


def normalize(a):
    """Normalize values in an array between `0` and `1`.

    Parameters
    ----------
    a : ndarray
        The array to be normalized.

    Returns
    -------
    out : ndarray
        The normalized array.
    """
    mi = a.min()
    mx = a.max()
    return (a - mi) / (mx - mi)
