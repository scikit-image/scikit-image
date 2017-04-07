"""
=======
Rescale
=======
The rescale operation resizes an image by a given scaling factor.
The scaling factor can either be a single floating point value,
or multiple values---one along each axis.

The Example takes the 'camera' image as input and rescales it by
a factor of 0.5.
"""


import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import rescale, resize, downscale_local_mean

image = data.camera()

rescale_image = rescale(image, 0.5)
resize_image = resize(image, (100, 100), mode='reflect')
image_downscale = downscale_local_mean(image, (3, 4))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 3),
                         sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})

ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].axis('off')
ax[0].set_title("Original Image")

ax[1].imshow(rescale_image, cmap='gray')
ax[1].axis('off')
ax[1].set_title("Rescaled Image")

ax[2].imshow(resize_image, cmap='gray')
ax[2].axis('off')
ax[2].set_title("Resized Image")

ax[3].imshow(image_downscale, cmap='gray')
ax[3].axis('off')
ax[3].set_title("Downscale Image using local averaging")

plt.show()
