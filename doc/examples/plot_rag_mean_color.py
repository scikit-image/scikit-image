"""
================
RAG Thresholding
================

This example constructs a Region Adjacency Graph (RAG) and merges regions
which are similar in color. We construct a RAG and define edges as the
difference in mean color. We then join regions with similar mean color.

"""

from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt


img = data.coffee()

labels1 = segmentation.slic(img, compactness=30, n_segments=400)
out1 = color.label2rgb(labels1, img, kind='avg')

g = graph.rag_mean_color(img, labels1)
labels2 = graph.cut_threshold(labels1, g, 29)
out2 = color.label2rgb(labels2, img, kind='avg')

plt.figure()
io.imshow(out1)
plt.figure()
io.imshow(out2)
io.show()
