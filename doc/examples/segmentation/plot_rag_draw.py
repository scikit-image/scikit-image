"""
======================================
Drawing Region Adjacency Graphs (RAGs)
======================================

This example constructs a Region Adjacency Graph (RAG) and draws it with
the `rag_draw` method.
"""
from skimage import data, segmentation
from skimage.future import graph
from matplotlib import pyplot as plt


img = data.coffee()
labels = segmentation.slic(img, compactness=30, n_segments=400)
g = graph.rag_mean_color(img, labels)


fig, ax = plt.subplots()
ax.set_title('RAG drawn will all edges')
graph.show_rag(labels, g, img, ax=ax)

fig, ax = plt.subplots()
ax.set_title('RAG drawn with edges having weight less than 30')
graph.show_rag(labels, g, img, thresh=30, ax=ax)

plt.show()
