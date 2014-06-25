"""
=======================
Region Adjacency Graphs
=======================

This example demonstrates the use of `merge_nodes` function of a Region
Adjacency Graph (RAG). When a new node is formed by merging two nodes, the edge
weight of all the edges incident on it can be updated by a user defined
function `weight_func`.

The default behaviour is to use the smaller edge weight incase of a conflict.
THe example below also shows how to use a custom function to take the larger
weight instead.

"""
from skimage.graph import rag
import networkx as nx
from matplotlib import pyplot as plt


def max_edge(g, src, dst, neighbor):
    try:
        w1 = g.edge[src][neighbor]['weight']
    except KeyError:
        w1 = None

    try:
        w2 = g.edge[dst][neighbor]['weight']
    except KeyError:
        w2 = None

    if w1 is None:
        return w2
    elif w2 is None:
        return w1
    else:
        return max(w1, w2)


def display(g, title):
    pos = nx.circular_layout(g)
    plt.figure()
    plt.title(title)
    nx.draw(g, pos)
    nx.draw_networkx_edge_labels(g, pos, font_size=20)


g = rag.RAG()
g.add_edge(1, 2, weight=10)
g.add_edge(2, 3, weight=20)
g.add_edge(3, 4, weight=30)
g.add_edge(4, 1, weight=40)
g.add_edge(1, 3, weight=50)

# Assigning dummy labels.
for n in g.nodes():
    g.node[n]['labels'] = [n]

gc = g.copy()

display(g, "Original Graph")

g.merge_nodes(1, 3)
display(g, "Merged with default (min)")

gc.merge_nodes(1, 3, weight_func=max_edge)
display(gc, "Merged with max")

plt.show()
