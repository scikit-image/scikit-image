"""
==============
Normalized Cut
==============

This example constructs a Region Adjacency Graph (RAG) and recursively performs
a Normalized Cut on it [1]_.

References
----------
.. [1] Shi, J.; Malik, J., "Normalized cuts and image segmentation",
       Pattern Analysis and Machine Intelligence,
       IEEE Transactions on, vol. 22, no. 8, pp. 888-905, August 2000.
"""
from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from numpy.random import seed
seed(9001)


img = data.coffee()

super_pixels = segmentation.slic(img, compactness=30, n_segments=400)
out1 = color.label2rgb(super_pixels, img, kind='avg')

g = graph.rag_mean_color(img, super_pixels, mode='similarity')

# Two ways to apply Normalized Cuts:

# Getting the value for a single threshold
labels1 = graph.cut_normalized(super_pixels, g, thresh=1e-3)

# Or using a generator to try several
label_gen = graph.cut_normalized_gen(super_pixels, g, thresh=1e-4)
labels2 = next(label_gen)
labels2 = label_gen.send(1e-3)

# Either method yields the same labels
assert (labels1 == labels2).all()


out2 = color.label2rgb(labels1, img, kind='avg')

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].imshow(out1)
ax[1].imshow(out2)

for a in ax:
    a.axis('off')

plt.tight_layout()
