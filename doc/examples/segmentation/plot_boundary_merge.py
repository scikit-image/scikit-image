from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np


def weight_boundary(graph, src, dst, n):
    if graph.has_edge(src, n) and graph.has_edge(dst, n):
        count_src = graph[src][n]['count']
        count_dst = graph[dst][n]['count']

        weight_src = graph[src][n]['weight']
        weight_dst = graph[dst][n]['weight']

        count = count_src + count_dst
        return {
            'count': count,
            'weight': (count_src*weight_src + count_dst*weight_dst)/count
        }

    elif graph.has_edge(src, n):
        return graph[src][n]
    elif graph.has_edge(dst, n):
        return graph[dst][n]


def merge_boundary(graph, src, dst):
    pass

img = data.coffee()
labels = segmentation.slic(img, compactness=30, n_segments=400)
g = graph.rag_mean_color(img, labels)

labels2 = graph.merge_hierarchical(labels, g, thresh=40, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_boundary,
                                   weight_func=weight_boundary)
