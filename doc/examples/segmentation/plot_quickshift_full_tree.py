"""
====================================================
Comparison of segmentation and superpixel algorithms
====================================================

This example compares the computation of a complete superpixel hierarchy
with the default which is to limit the parent to a local neighborhood
of each pixel. When using ``full_search=True``the algorithm will return
a single tree for the whole image provided that ``max_dist`` is large enough
or that ``return_tree=True``.

Since computing a full hierarchy requires searching the
whole image it is also slower.


Quickshift image segmentation
-----------------------------

Quickshift is a relatively recent 2D image segmentation algorithm, based on an
approximation of kernelized mean-shift. Therefore it belongs to the family of
local mode-seeking algorithms and is applied to the 5D space consisting of
color information and image location [2]_.

One of the benefits of quickshift is that it actually computes a
hierarchical segmentation on multiple scales simultaneously.

Quickshift has two main parameters: ``sigma`` controls the scale of the local
density approximation, ``max_dist`` selects a level in the hierarchical
segmentation that is produced. There is also a trade-off between distance in
color-space and distance in image-space, given by ``ratio``.

.. [2] Quick shift and kernel methods for mode seeking,
    Vedaldi, A. and Soatto, S.
    European Conference on Computer Vision, 2008

"""

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.segmentation import quickshift
from skimage.util import img_as_float

img = img_as_float(astronaut()[::2, ::2])

# compute a complete tree of the image
segments_quick = quickshift(img, kernel_size=3, max_dist=1000,
                            ratio=0.5, return_tree=True, full_search=True)
top_level = segments_quick[0]
parents = np.array(segments_quick[1]).astype('uint32')
dist_to_parent = np.array(segments_quick[2]).astype('uint32')

segments_quick_partial = quickshift(img, kernel_size=3, max_dist=1000,
                                    ratio=0.5, return_tree=True,
                                    full_search=False)
top_level_partial = segments_quick_partial[0]
parents_partial = np.array(segments_quick_partial[1]).astype('uint32')
dist_to_parent_partial = np.array(segments_quick_partial[2]).astype('uint32')

fig, ax = plt.subplots(3, 2, figsize=(20, 20), sharex=True, sharey=True)

ax[0, 0].imshow(segments_quick[0])
ax[0, 0].set_title("top level full search")
ax[1, 0].imshow(np.array(segments_quick[1]).astype('uint32'))
ax[1, 0].set_title('parents full search')
ax[2, 0].imshow(np.array(segments_quick[2]).astype('uint32'))
ax[2, 0].set_title('distance to parent full search')


ax[0, 1].imshow(segments_quick_partial[0])
ax[0, 1].set_title("top level partial search")
ax[1, 1].imshow(np.array(segments_quick_partial[1]).astype('uint32'))
ax[1, 1].set_title('parents partial search')
ax[2, 1].imshow(np.array(segments_quick_partial[2]).astype('uint32'))
ax[2, 1].set_title('distance to parent partial search')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()


# perform multilevel segmentation
def multilevel_segment(parents):
    parents_display = parents.copy()
    parents_display_old = np.zeros_like(parents_display)
    ims = []
    nb_levels = 0
    while np.any(parents_display_old != parents_display):
        parents_display_old = parents_display.copy()
        parents_display = np.squeeze(parents_display[
                                            np.unravel_index([parents_display],
                                            parents_display.shape)])
        ims.append(parents_display)
        nb_levels = nb_levels + 1
    nb_levels = nb_levels - 1
    return ims[:-1], nb_levels

ims, nb_levels = multilevel_segment(parents)
ims_partial, nb_levels_partial = multilevel_segment(parents_partial)

plt.set_cmap('viridis')
fig, ax = plt.subplots(max(nb_levels, nb_levels_partial), 2, figsize=(20,
                       max(nb_levels, nb_levels_partial) * 10),
                       sharex=True, sharey=True)
for i in range(nb_levels):
    ax[i, 0].imshow(ims[i])
    ax[i, 0].set_title("full search level " + str(i))
for i in range(nb_levels_partial):
    ax[i, 1].imshow(ims_partial[i])
    ax[i, 1].set_title("partial search level " + str(i))

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()


# perform multiscale segmentation
def multiscale_segment(segments_quick, nb_scales, min_scale):
    ims = []
    last_scale = 0
    for i in range(min_scale, nb_scales + min_scale, 1):
        max_dist = 2**i
        # remove parents with distance > max_dist
        parent_flat = np.array(segments_quick[1]).astype('uint32').ravel()
        old = np.zeros(parent_flat.shape).astype('uint32')
        too_far = np.array(segments_quick[2]).ravel() > max_dist
        parent_flat[too_far] = np.arange(parent_flat.shape[0])[too_far]
        while np.any(old != parent_flat):
            old = parent_flat
            parent_flat = parent_flat[parent_flat]
        ims.append(np.reshape(parent_flat, segments_quick[0].shape))
        if len(ims) >= 2:
            if np.any(ims[-1] != ims[last_scale-1]) or last_scale == 0:
                last_scale = len(ims)
    return ims[:last_scale]
nb_scales = 8
min_scale = 2
ims = multiscale_segment(segments_quick, nb_scales, min_scale)
ims_partial = multiscale_segment(segments_quick_partial, nb_scales, min_scale)

plt.set_cmap('viridis')
fig, ax = plt.subplots(max(len(ims), len(ims_partial)), 2,
                       figsize=(20, max(len(ims), len(ims_partial)) * 10),
                       sharex=True, sharey=True)
for i in range(len(ims)):
    ax[i, 0].imshow(ims[i])
    ax[i, 0].set_title("full search scale " + str(2**(min_scale+i)))
for i in range(len(ims_partial)):
    ax[i, 1].imshow(ims_partial[i])
    ax[i, 1].set_title("partial search scale " + str(2**(min_scale+i)))

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()
