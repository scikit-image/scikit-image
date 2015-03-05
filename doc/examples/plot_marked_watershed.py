"""
===============================
Markers for watershed transform
===============================

The watershed is a classical algorithm used for **segmentation**, that
is, for separating different objects in an image.

Here a marker image is build from the region of low gradient inside the image.

See Wikipedia_ for more details on the algorithm.

.. _Wikipedia: http://en.wikipedia.org/wiki/Watershed_(image_processing)

"""

from scipy import ndimage
import matplotlib.pyplot as plt

from skimage.morphology import watershed, disk
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte


image = img_as_ubyte(data.camera())

# denoise image
denoised = rank.median(image, disk(2))

# find continuous region (low gradient) --> markers
markers = rank.gradient(denoised, disk(5)) < 10
markers = ndimage.label(markers)[0]

#local gradient
gradient = rank.gradient(denoised, disk(2))

# process the watershed
labels = watershed(gradient, markers)

# display results
fig, axes = plt.subplots(ncols=4, figsize=(8, 2.7))
ax0, ax1, ax2, ax3 = axes

ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax1.imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
ax2.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
ax3.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax3.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)

for ax in axes:
    ax.axis('off')

fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
plt.show()
