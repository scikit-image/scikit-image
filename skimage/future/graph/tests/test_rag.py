import numpy as np
from skimage.future import graph
from skimage._shared.version_requirements import is_installed
from numpy.testing.decorators import skipif
from skimage import segmentation
from numpy import testing


def max_edge(g, src, dst, n):
    default = {'weight': -np.inf}
    w1 = g[n].get(src, default)['weight']
    w2 = g[n].get(dst, default)['weight']
    return max(w1, w2)


@skipif(not is_installed('networkx'))
def test_rag_merge():
    g = graph.rag.RAG()

    for i in range(5):
        g.add_node(i, {'labels': [i]})

    g.add_edge(0, 1, {'weight': 10})
    g.add_edge(1, 2, {'weight': 20})
    g.add_edge(2, 3, {'weight': 30})
    g.add_edge(3, 0, {'weight': 40})
    g.add_edge(0, 2, {'weight': 50})
    g.add_edge(3, 4, {'weight': 60})

    gc = g.copy()

    # We merge nodes and ensure that the minimum weight is chosen
    # when there is a conflict.
    g.merge_nodes(0, 2)
    assert g.edge[1][2]['weight'] == 10
    assert g.edge[2][3]['weight'] == 30

    # We specify `max_edge` as `weight_func` as ensure that maximum
    # weight is chosen in case on conflict
    gc.merge_nodes(0, 2, weight_func=max_edge)
    assert gc.edge[1][2]['weight'] == 20
    assert gc.edge[2][3]['weight'] == 40

    g.merge_nodes(1, 4)
    g.merge_nodes(2, 3)
    n = g.merge_nodes(3, 4, in_place=False)
    assert sorted(g.node[n]['labels']) == list(range(5))
    assert g.edges() == []


@skipif(not is_installed('networkx'))
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

    rag = graph.rag_mean_color(img, labels)
    new_labels = graph.cut_threshold(labels, rag, 10, in_place=False)
    # Two labels
    assert new_labels.max() == 1

    new_labels = graph.cut_threshold(labels, rag, 10)
    # Two labels
    assert new_labels.max() == 1


@skipif(not is_installed('networkx'))
def test_cut_normalized():

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

    rag = graph.rag_mean_color(img, labels, mode='similarity')

    new_labels = graph.cut_normalized(labels, rag, in_place=False)
    new_labels, _, _ = segmentation.relabel_sequential(new_labels)
    # Two labels
    assert new_labels.max() == 1

    new_labels = graph.cut_normalized(labels, rag)
    new_labels, _, _ = segmentation.relabel_sequential(new_labels)
    assert new_labels.max() == 1


@skipif(not is_installed('networkx'))
def test_rag_error():
    img = np.zeros((10, 10, 3), dtype='uint8')
    labels = np.zeros((10, 10), dtype='uint8')
    labels[:5, :] = 0
    labels[5:, :] = 1
    testing.assert_raises(ValueError, graph.rag_mean_color, img, labels,
                          2, 'non existant mode')


def _weight_mean_color(graph, src, dst, n):
    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return diff


def _pre_merge_mean_color(graph, src, dst):
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])


def merge_hierarchical_mean_color(labels, rag, thresh, rag_copy=True,
                                  in_place_merge=False):
    return graph.merge_hierarchical(labels, rag, thresh, rag_copy,
                                    in_place_merge, _pre_merge_mean_color,
                                    _weight_mean_color)


@skipif(not is_installed('networkx'))
def test_rag_hierarchical():
    img = np.zeros((8, 8, 3), dtype='uint8')
    labels = np.zeros((8, 8), dtype='uint8')

    img[:, :, :] = 31
    labels[:, :] = 1

    img[0:4, 0:4, :] = 10, 10, 10
    labels[0:4, 0:4] = 2

    img[4:, 0:4, :] = 20, 20, 20
    labels[4:, 0:4] = 3

    g = graph.rag_mean_color(img, labels)
    g2 = g.copy()
    thresh = 20  # more than 11*sqrt(3) but less than

    result = merge_hierarchical_mean_color(labels, g, thresh)
    assert(np.all(result[:, :4] == result[0, 0]))
    assert(np.all(result[:, 4:] == result[-1, -1]))

    result = merge_hierarchical_mean_color(labels, g2, thresh,
                                           in_place_merge=True)
    assert(np.all(result[:, :4] == result[0, 0]))
    assert(np.all(result[:, 4:] == result[-1, -1]))

    result = graph.cut_threshold(labels, g, thresh)
    assert np.all(result == result[0, 0])
