"""
==========================
Region Boundary based RAGs
==========================

This example demonstrates construction of region boundary based RAGs with the
`rag_boundary` function.
"""
from skimage.future import graph
from skimage import data, segmentation, color, filters, io

img = data.coffee()
gimg = color.rgb2gray(img)

labels = segmentation.slic(img, compactness=30, n_segments=400)
edges = filters.sobel(gimg)
edges_rgb = color.gray2rgb(edges)

g = graph.rag_boundary(labels, edges)

out = graph.draw_rag(labels, g, edges_rgb, node_color="#ffff00",
                     colormap='jet')

io.imshow(out)
io.show()
