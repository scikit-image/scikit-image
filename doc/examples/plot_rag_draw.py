"""
=====================================
Drawing Region Adjacency Graphs (RAGs)
======================================

This example constructs a Region Adjacency Graph (RAG) and draws it with
the `rag_draw` method.
"""
from skimage import graph, data, segmentation
from matplotlib import pyplot as plt, colors


img = data.coffee()
labels = segmentation.slic(img, compactness=30, n_segments=400)
g = graph.rag_mean_color(img, labels)

out = graph.draw_rag(labels, g, img)
plt.figure()
plt.title("RAG with all edges shown in green.")
plt.imshow(out)

cmap = colors.ListedColormap(['#00ff00', '#ff0000'])
out = graph.draw_rag(labels, g, img, colormap=cmap, thresh=30, desaturate=True)
plt.figure()
plt.title("RAG with edge weights less than 30, color "
          "mapped between green and red.")

plt.imshow(out)
plt.show()
