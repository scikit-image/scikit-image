"""
========================================
Splitting clumped objects with watershed
========================================

Thresholding is often the first step of a segmentation workflow: the pixels
that belong to an object are separated from the background. Thresholding
does not, however, tell touching objects apart, because they end up in the
same connected region.

In this example, we separate such clumps with a seeded watershed. The
distance transform of the binary mask gives every object pixel its distance
to the nearest background pixel, so the center of an object is a local
maximum of the distance. Markers placed at these maxima let the watershed
grow each object from its own center, even where objects share an outline.

We use an image of human cell nuclei provided by Jason Moffat [1]_ through
`CellProfiler <https://cellprofiler.org/examples/#human-cells>`_.

.. [1] Moffat J, Grueneberg DA, Yang X, Kim SY, Kloepfer AM, Hinkle G,
       Piqani B, Eisenhaure TM, Luo B, Grenier JK, Carpenter AE, Foo SY,
       Stewart SA, Stockwell BR, Hacohen N, Hahn WC, Lander ES, Sabatini DM,
       Root DE (2006) "A lentiviral RNAi library for human and mouse genes
       applied to an arrayed viral high-content screen" Cell, 124(6):1283-98.
       PMID: 16564017
       :DOI:`10.1016/j.cell.2006.01.040`

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import skimage as ski

#####################################################################
# Threshold the image
# ===================
# The nuclei are bright on a dark background, so a global threshold yields a
# reasonable binary mask. Its connected components are the objects found by
# thresholding alone.

image = ski.data.human_mitosis()

mask = image > ski.filters.threshold_otsu(image)
labeled_mask = ski.measure.label(mask)
print(f'Objects after thresholding: {labeled_mask.max()}')

distance = ndi.distance_transform_edt(mask)

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(mask, cmap='gray')
axes[1].set_title(f'Thresholded ({labeled_mask.max()} objects)')
axes[2].imshow(distance, cmap='magma')
axes[2].set_title('Distance transform')
for ax in axes:
    ax.set_axis_off()
fig.tight_layout()
plt.show()

#####################################################################
# Find the markers
# ================
# The centers of the objects are the local maxima of the distance transform.
# The nuclei are about ten pixels wide, so we require neighboring maxima to
# lie at least five pixels apart. We also keep maxima near the image border
# (``exclude_border=False``), where some nuclei are cut off.

peak_coords = ski.feature.peak_local_max(
    distance, min_distance=5, exclude_border=False, labels=mask
)
marker_mask = np.zeros(distance.shape, dtype=bool)
marker_mask[tuple(peak_coords.T)] = True
markers = ski.measure.label(marker_mask)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(distance, cmap='magma')
ax.plot(peak_coords[:, 1], peak_coords[:, 0], 'r.')
ax.set_title('Distance transform and markers')
ax.set_axis_off()
plt.show()

#####################################################################
# Split the objects
# =================
# The watershed floods the image starting from the markers. We pass the
# negative distance transform so that the flooding starts at the object
# centers, and the binary mask so that the labels stay inside the objects.

labels = ski.segmentation.watershed(-distance, markers, mask=mask)
print(f'Objects after watershed: {labels.max()}')

#####################################################################
# The two segmentations below make the difference visible: nuclei that end
# up with a single label after thresholding receive distinct labels after
# the watershed, so the number of objects increases.

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(ski.color.label2rgb(labeled_mask, bg_label=0))
axes[0].set_title(f'Connected components ({labeled_mask.max()})')
axes[1].imshow(ski.color.label2rgb(labels, bg_label=0))
axes[1].set_title(f'Watershed ({labels.max()})')
for ax in axes:
    ax.set_axis_off()
fig.tight_layout()
# sphinx_gallery_thumbnail_number = 3
plt.show()
