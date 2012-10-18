"""
===================
Label image regions
===================

This example shows how to segment an image with image labelling. The following
steps are applied:

1. Thresholding with automatic Otsu method
2. Close small holes with binary closing
3. Remove artifacts touching image border
4. Measure image regions to filter small objects

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filter import threshold_otsu

from skimage.filter.rank import modal

from skimage.morphology import label, disk
from skimage.measure import find_contours


image = data.coins()[50:-50, 50:-50]

# apply threshold
thresh = threshold_otsu(image)
bw = image > thresh

# label image regions
label_image = label(bw)

# filter obtained labels using model filter
mod_label_image = modal(label_image.astype(np.uint16),disk(5))

# the background is here 1
contours = find_contours(mod_label_image==1,0, positive_orientation='low')

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6, 6))

print axes

ax0, ax1, ax2, ax3 = axes.ravel()

ax0.imshow(bw, cmap='gray')
ax0.set_title('Otsu threshold')
ax1.imshow(label_image, cmap='jet')
ax1.set_title('label image')
ax2.imshow(mod_label_image, cmap='jet')
ax2.set_title('filtered labels (modal)')
ax3.imshow(image, cmap='gray')
ax3.set_title('contour overlay')
ax3.set_xlim((0,image.shape[1]))
ax3.set_ylim((image.shape[0],0))


for n, contour in enumerate(contours):
    ax3.plot(contour[:, 1], contour[:, 0], linewidth=2)


plt.show()

