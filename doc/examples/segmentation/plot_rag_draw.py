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
ax.set_title('RAG drawn with default settings')
lc = graph.show_rag(labels, g, img, ax=ax)
# fraction specifies the fraction of the area of the plot that will be used to
# draw the colorbar
plt.colorbar(lc, fraction=0.03)

fig, ax = plt.subplots()
ax.set_title('RAG drawn with grayscale image and viridis colormap')
lc = graph.show_rag(labels, g, img, img_cmap='gray', edge_cmap='viridis',
                    ax=ax)
plt.colorbar(lc, fraction=0.03)

plt.show()
