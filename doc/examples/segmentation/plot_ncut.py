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


img = data.coffee()

labels0 = segmentation.slic(img, compactness=30, n_segments=400)
out0 = color.label2rgb(labels0, img, kind='avg')

g = graph.rag_mean_color(img, labels0, mode='similarity')

# Two ways to apply Normalized Cuts:
# Get the labels for a single threshold
labels1a = graph.cut_normalized(labels0, g, thresh=1e-3)

# Or use a generator to try several thresholds
label_gen = graph.cut_normalized_gen(labels0, g, init_thresh=1e-4)
labels2 = next(label_gen)
labels1b = label_gen.send(1e-3)


out1a = color.label2rgb(labels1a, img, kind='avg')
out2 = color.label2rgb(labels2, img, kind='avg')
out1b = color.label2rgb(labels1b, img, kind='avg')

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                       figsize=(8, 6))
ax = ax.ravel()

ax[0].imshow(out0)
ax[0].set_title('SLIC')
ax[1].imshow(out1a)
ax[1].set_title('N-cut threshold: 1e-3')
ax[2].imshow(out2)
ax[2].set_title('N-cut generator threshold: 1e-4')
ax[3].imshow(out1b)
ax[3].set_title('N-cut generator threshold: 1e-3')

for a in ax:
    a.axis('off')

plt.tight_layout()
