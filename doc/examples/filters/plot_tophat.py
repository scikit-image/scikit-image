"""
================================================================
Removing small objects in grayscale images with a top hat filter
================================================================

This example shows how to remove small objects from grayscale images.
The top-hat transform [1]_ extracts small elements and details from an image.
Here, the white top-hat transform extracts bright structures smaller than the
footprint. Morphological opening produces the complementary result, with these
bright structures removed.

.. [1] https://en.wikipedia.org/wiki/Top-hat_transform

"""

import matplotlib.pyplot as plt

from skimage import data
from skimage import color, morphology

image = color.rgb2gray(data.hubble_deep_field())[:500, :500]

footprint = morphology.disk(1)
white_tophat = morphology.white_tophat(image, footprint)
opened = morphology.opening(image, footprint)

fig, ax = plt.subplots(ncols=3, figsize=(20, 8))
ax[0].set_title('Original')
ax[0].imshow(image, cmap='gray')
ax[1].set_title('White top-hat')
ax[1].imshow(white_tophat, cmap='gray')
ax[2].set_title('Morphological opening')
ax[2].imshow(opened, cmap='gray')

plt.show()
