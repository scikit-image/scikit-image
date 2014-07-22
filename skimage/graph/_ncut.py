import networkx as nx
import numpy as np
from scipy import sparse


def DW_matrix(graph):

    W = nx.to_scipy_sparse_matrix(graph, format='csc')
    entries = W.sum(0)
    D = sparse.dia_matrix((entries, 0), shape=W.shape).tocsc()
    return D, W


def ncut_cost(mask, D, W):

    mask = np.array(mask)
    mask_list = [np.logical_xor(mask[i], mask) for i in range(mask.shape[0])]
    mask_array = np.array(mask_list)

    cut = float(W[mask_array].sum() / 2.0)
    assoc_a = D.data[mask].sum()
    assoc_b = D.data[np.logical_not(mask)].sum()

    return (cut / assoc_a) + (cut / assoc_b)


def normalize(a):
    mi = a.min()
    mx = a.max()
    return (a - mi) / (mx - mi)
