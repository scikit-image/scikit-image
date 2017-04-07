"""
=======
Rescale
=======
The rescale operation resizes an image by a given scaling factor.
The scaling factor can either be a single floating point value,
or multiple values---one along each axis.

The Example takes a camera image as input and rescales it by a
factor of 0.5.
"""


import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import rescale, resize, downscale_local_mean

image = data.camera()

rescale_image = rescale(image, 0.5)

resize_image = resize(image, (100, 100), mode='reflect')

image_downscale = downscale_local_mean(image, (3, 4))

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                                         sharex=True, sharey=True,
                                         subplot_kw={'adjustable': 'box-forced'})

ax0.imshow(image, cmap=plt.cm.gray)
ax0.axis('off')
ax0.set_title("Original Image")

ax1.imshow(rescale_image, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title("Rescaled Image")

ax2.imshow(resize_image, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title("Resized Image")

ax2.imshow(image_downscale, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title("Downscale Image using local averaging")

plt.show()
