"""
==============================
Rescale, resize, and downscale
==============================

`Rescale` operation resizes an image by a given scaling factor.
The scaling factor can either be a single floating point value,
or multiple values - one along each axis.

`Resize` serves the same purpose, but allows to specify an output
image shape instead of a scaling factor.

`Downscale` operation serves the purpose of downsampling an
n-dimensional image by calculating local mean on the elements of
each block of the size factors given as a parameter to the function.
"""

import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import rescale, resize, downscale_local_mean

image = data.camera()

image_rescaled = rescale(image, 0.5)
image_resized = resize(image, (400, 400), mode='reflect')
image_downscaled = downscale_local_mean(image, (2, 3))

fig, axes = plt.subplots(nrows=2, ncols=2,
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original image")

ax[1].imshow(image_rescaled, cmap='gray')
ax[1].set_title("Rescaled image")

ax[2].imshow(image_resized, cmap='gray')
ax[2].set_title("Resized image")

ax[3].imshow(image_downscaled, cmap='gray')
ax[3].set_title("Image downscaled using local averaging")

plt.tight_layout()
plt.show()
