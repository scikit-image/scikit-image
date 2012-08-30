"""
===================
Label image regions
===================

This example shows how to segment an image with image labelling.

"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops


image = data.coins()[50:-50, 50:-50]

# apply threshold
thresh = threshold_otsu(image)
bw = image > thresh

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
plt.gray()
ax.imshow(bw)

# remove artifacts connected to image border
cleared = bw.copy()
clear_border(cleared)

# label image regions
label_image = label(cleared)

for region in regionprops(label_image, ['Area', 'BoundingBox']):

    # skip small images
    if region['Area'] < 100:
        continue

    # draw rectangle around segmented coins
    minr, minc, maxr, maxc = region['BoundingBox']
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

plt.show()
