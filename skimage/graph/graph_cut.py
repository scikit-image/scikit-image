import networkx as nx
import numpy as np

def threshold_cut(label, rag, thresh):

    #print [rag.edges_iter(data = True)]
    to_remove = [(x,y) for x,y,d in rag.edges_iter(data = True) if d['weight'] >= thresh]
    #print "edges to remove",len(to_remove)

    rag.remove_edges_from(to_remove)

    #print "to remove", to_remove

    comps = nx.connected_components(rag)
    out = np.copy(label)
    #print "comps",len(comps)

    for i, nodes in enumerate(comps) :
        
        for node in nodes :
            for l in rag.node[node]['labels'] :
                out[label == l] = i

    #print out
    #print label
    return out
