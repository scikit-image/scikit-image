"""
===============================================================
Comparing edge-based segmentation and region-based segmentation
===============================================================

In this example, we will see how to segment objects from a background.
We use the ``coins`` image from ``skimage.data``. This image shows
several coins outlined against a darker background. The segmentation of
the coins cannot be done directly from the histogram of grey values,
because the background shares enough grey levels with the coins that a
thresholding segmentation is not sufficient. Simply thresholding the image 
leads either to missing significant parts of the coins, or to merging parts
of the background with the coins.

We first try an edge-based segmentation. We use the Canny detector to 
delineate the contours of the coins. These contours are filled using 
mathematical morphology (``scipy.ndimage.binary_fill_holes``). Small spurious 
objects are easily removed by applying a threshold on the size of 
unconnected objects. However, this method is not very robust, since contours 
that are not perfectly closed are not filled correctly. This happens for one 
of the coins.

We therefore try a second method, that is region-based. Here we use the 
watershed transform. An elevation map is provided by the Sobel gradient
of the image. Markers of the background and the coins are determined from
the extreme parts of the histogram of grey values.

This second method works even better, and the coins can be segmented and 
labeled individually.

"""


import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import skimage
from skimage.filter import canny, sobel
from skimage.morphology import watershed

#------------------ Loading data --------------------------------
from skimage import data
coins = data.coins()

#------------ Histogram of grey values ---------------------------
histo = np.histogram(coins, bins=np.arange(0, 256))

plt.figure(figsize=(8, 3))
plt.subplot(121)
plt.imshow(coins, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.subplot(122)
plt.plot(histo[1][:-1], histo[0], lw=2)
plt.title('histogram of grey values')

#------------------ Tentative thresholding --------------------------------
plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.imshow(coins > 100, cmap=plt.cm.gray, interpolation='nearest')
plt.title('coins > 100')
plt.axis('off')
plt.subplot(122)
plt.imshow(coins > 150, cmap=plt.cm.gray, interpolation='nearest')
plt.title('coins > 150')
plt.axis('off')

plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)


#------------------ Edge-based segmentation --------------------------------
edges = canny(coins/255.)

fill_coins = ndimage.binary_fill_holes(edges)

label_objects, nb_labels = ndimage.label(fill_coins)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0
coins_cleaned = mask_sizes[label_objects]

plt.figure(figsize=(7, 3.))
plt.subplot(131)
plt.imshow(edges, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('Canny detector')
plt.subplot(132)
plt.imshow(fill_coins, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('Filling the holes')
plt.subplot(133)
plt.imshow(coins_cleaned, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('Removing small objects')
plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)


#------------------ Region-based segmentation --------------------------------

markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2

elevation_map = sobel(coins)


segmentation = watershed(elevation_map, markers)


plt.figure(figsize=(7, 3))
plt.subplot(131)
plt.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
plt.axis('off')
plt.title('markers')
plt.subplot(132)
plt.imshow(elevation_map, cmap=plt.cm.jet, interpolation='nearest')
plt.axis('off')
plt.title('elevation_map')
plt.subplot(133)
plt.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('segmentation')

plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)

# ------------------- Removing a few small holes ---------------------

segmentation = ndimage.binary_fill_holes(segmentation - 1)

#------------------ Labeling the coins --------------------------------
labeled_coins, _ = ndimage.label(segmentation)

plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.imshow(coins, cmap=plt.cm.gray, interpolation='nearest')
plt.contour(segmentation, [0.5], linewidths=1.2, colors='y')
plt.axis('off')
plt.subplot(122)
plt.imshow(labeled_coins, cmap=plt.cm.spectral, interpolation='nearest')
plt.axis('off')

plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)


plt.show()
