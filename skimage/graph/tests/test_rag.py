import numpy as np
from skimage import graph
import random

def _min_edge(e1,e2,g):
    w1 = g.edge[e1[0]][e1[1]]['weight']
    w2 = g.edge[e2[0]][e2[1]]['weight']
    return min(w1,w2)

def test_rag_merge():
    g = graph.rag.RAG()
    for i in range(10):
        g.add_edge(i, (i + 1) % 10, {'weight': i * 10})
        g.node[i]['labels'] = [i]

    for i in range(4):
        x = random.choice(g.nodes())
        y = random.choice(g.nodes())
        while x == y :
            y = random.choice(g.nodes())
        g.merge_nodes(x,y)
            
    for i in range(5):
        x = random.choice(g.nodes())
        y = random.choice(g.nodes())
        while x == y :
            y = random.choice(g.nodes())
        g.merge_nodes(x,y,_min_edge)

    idx = g.nodes()[0]
    assert sorted(g.node[idx]['labels']) == list(range(10))
    assert g.edges() == []


def test_threshold_cut():

    img = np.zeros((100, 100, 3), dtype='uint8')
    img[:50, :50] = 255, 255, 255
    img[:50, 50:] = 254, 254, 254
    img[50:, :50] = 2, 2, 2
    img[50:, 50:] = 1, 1, 1

    labels = np.zeros((100, 100), dtype='uint8')
    labels[:50, :50] = 0
    labels[:50, 50:] = 1
    labels[50:, :50] = 2
    labels[50:, 50:] = 3

    rag = graph.rag_meancolor(img, labels)
    new_labels = graph.threshold_cut(labels, rag, 10)

    # Two labels
    assert new_labels.max() == 1
