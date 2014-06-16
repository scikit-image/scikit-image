"""
================
RAG Thresholding
================

This examples constructs a Region Adjacency Graph and merges region which are
similar in color. We construct a RAG and define edges as the difference in
mean color. We the join regions with similar mean color.

"""

from skimage import graph
from skimage import segmentation
from skimage import data, io
import numpy as np
from matplotlib import pyplot as plt


def label_mask_img(img, label):

    out = np.zeros_like(img)

    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    for i in range(label.max()):
        mask = label == i

        r = np.average(red[mask])
        g = np.average(green[mask])
        b = np.average(blue[mask])

        # print r,g,b
        out[mask] = r, g, b

    return out

img = data.coffee()

labels1 = segmentation.slic(img, compactness=30, n_segments=400)
out1 = label_mask_img(img, labels1)

g = graph.rag_meancolor(img, labels1)
labels2 = graph.threshold_cut(labels1, g, 30)
out2 = label_mask_img(img, labels2)

plt.figure()
io.imshow(out1)
plt.figure()
io.imshow(out2)
io.show()
