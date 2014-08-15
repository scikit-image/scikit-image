"""
===========
RAG Merging
===========

This example constructs a Region Adjacency Graph (RAG) and progressively merges
regions that are similar in color. Merging two adjacent regions produces
a new regions with all the pixels from the merged regions. Regions are merged
until no highly similar regions remain.

"""

from skimage import graph, data, io, segmentation, color


img = data.coffee()
labels = segmentation.slic(img, compactness=30, n_segments=400)
g = graph.rag_mean_color(img, labels)
labels2 = graph.merge_hierarchical(labels, g, 40)
g2 = graph.rag_mean_color(img, labels2)

out = color.label2rgb(labels2, img, kind='avg')
out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
io.imshow(out)
io.show()
