"""
===============================================================
Comparing edge-based segmentation and region-based segmentation
===============================================================

In this example, we will see how to segment objects from a background.  We use
the ``coins`` image from ``skimage.data``, which shows several coins outlined
against a darker background.
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data

coins = data.coins()
hist = np.histogram(coins, bins=np.arange(0, 256))

plt.figure(figsize=(8, 3))
plt.subplot(121)
plt.imshow(coins, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.subplot(122)
plt.plot(hist[1][:-1], hist[0], lw=2)
plt.title('histogram of grey values')

"""
.. image:: PLOT2RST.current_figure

Thresholding
============

A simple way to segment the coins is to choose a threshold based on the
histogram of grey values. Unfortunately, thresholding this image gives a binary
image that either misses significant parts of the coins or merges parts of the
background with the coins:

"""

plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.imshow(coins > 100, cmap=plt.cm.gray, interpolation='nearest')
plt.title('coins > 100')
plt.axis('off')
plt.subplot(122)
plt.imshow(coins > 150, cmap=plt.cm.gray, interpolation='nearest')
plt.title('coins > 150')
plt.axis('off')
margins = dict(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
plt.subplots_adjust(**margins)

"""
.. image:: PLOT2RST.current_figure


Edge-based segmentation
=======================

Next, we try to delineate the contours of the coins using edge-based
segmentation. To do this, we first get the edges of features using the Canny
edge-detector.
"""

from skimage.filter import canny
edges = canny(coins/255.)

plt.figure(figsize=(4, 3))
plt.imshow(edges, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('Canny detector')

"""
.. image:: PLOT2RST.current_figure

These contours are then filled using mathematical morphology.
"""

from scipy import ndimage

fill_coins = ndimage.binary_fill_holes(edges)

plt.figure(figsize=(4, 3))
plt.imshow(fill_coins, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('Filling the holes')

"""
.. image:: PLOT2RST.current_figure

Small spurious objects are easily removed by setting a minimum size for valid
objects.
"""

label_objects, nb_labels = ndimage.label(fill_coins)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0
coins_cleaned = mask_sizes[label_objects]

plt.figure(figsize=(4, 3))
plt.imshow(coins_cleaned, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('Removing small objects')

"""
.. image:: PLOT2RST.current_figure

However, this method is not very robust, since contours that are not perfectly
closed are not filled correctly, as is the case for one unfilled coin above.


Region-based segmentation
=========================

We therefore try a region-based method using the
watershed transform. First, we find an elevation map using the Sobel gradient of the
image. 

"""

from skimage.filter import sobel

elevation_map = sobel(coins)

plt.figure(figsize=(4, 3))
plt.imshow(elevation_map, cmap=plt.cm.jet, interpolation='nearest')
plt.axis('off')
plt.title('elevation_map')

"""
.. image:: PLOT2RST.current_figure

Next we find markers of the background and the coins based on the extreme parts
of the histogram of grey values.
"""

markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2

plt.figure(figsize=(4, 3))
plt.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
plt.axis('off')
plt.title('markers')

"""
.. image:: PLOT2RST.current_figure

Finally, we use the watershed transform to fill regions of the elevation map starting from the markers determined above:

"""
from skimage.morphology import watershed
segmentation = watershed(elevation_map, markers)

plt.figure(figsize=(4, 3))
plt.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('segmentation')

"""
.. image:: PLOT2RST.current_figure

This last method works even better, and the coins can be segmented and 
labeled individually.

"""

segmentation = ndimage.binary_fill_holes(segmentation - 1)
labeled_coins, _ = ndimage.label(segmentation)

plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.imshow(coins, cmap=plt.cm.gray, interpolation='nearest')
plt.contour(segmentation, [0.5], linewidths=1.2, colors='y')
plt.axis('off')
plt.subplot(122)
plt.imshow(labeled_coins, cmap=plt.cm.spectral, interpolation='nearest')
plt.axis('off')

plt.subplots_adjust(**margins)

"""
.. image:: PLOT2RST.current_figure

"""

plt.show()
