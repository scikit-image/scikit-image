import networkx as nx
import numpy as np
from scipy import sparse

def DW_matrix(graph):
    #print graph[4][4]['weight']
    W = nx.to_scipy_sparse_matrix(graph, format='csc')
    entries = W.sum(0)
    D = sparse.dia_matrix( (entries,0),shape = W.shape).tocsc()
    #print W[4,4]
    
    m,n = W.shape
    #for i in range(n):
    #    W[i,i] = 1.0
    return D,W

def ncut_cost(mask,D,W):

    mask = np.array(mask)
    mask_list = [ np.logical_xor(mask[i], mask) for i in range(mask.shape[0])]
    mask_array = np.array(mask_list)
       
    cut = float(W[mask_array].sum()/2.0)
    #print W.todense()
    #print mask_array.astype(int)
    #print "cut=",cut
    
    assoc_a = D.data[mask].sum()
    assoc_b = D.data[np.logical_not(mask)].sum()
    
    #print cut
    #print assoc_a,assoc_b
    return (cut/assoc_a) + (cut/assoc_b)

def norml(a):
    mi = a.min()
    mx = a.max()
    return (a-mi)/(mx-mi)
