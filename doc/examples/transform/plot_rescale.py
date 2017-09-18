"""
==============================
Rescale, resize, and downscale
==============================

`Rescale` operation resizes an image by a given scaling factor. The scaling
factor can either be a single floating point value, or multiple values - one
along each axis.

`Resize` serves the same purpose, but allows to specify an output image shape
instead of a scaling factor.

`Downscale` operation serves the purpose of downsampling an n-dimensional image
by Gaussian smoothing and by calculating local mean on the elements of each
block of the size factors given as a parameter to the function.

Note that when downsampling an image, `resize` and `rescale` might produce
aliasing effects since they are not appropriately resampling the input image.
Please, refer to the `downsize`, `downscale`, and `downscale_local_mean`
functions in this case.

"""

import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import rescale, resize, downscale, downscale_local_mean

image = data.camera()

image_rescaled = rescale(image, 0.5)
image_resized = resize(image, (400, 400), mode='reflect')
image_downscaled = downscale(image, 2)
image_downscaled_local_mean = downscale_local_mean(image, (2, 2))

fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original image")

ax[1].imshow(image_rescaled, cmap='gray')
ax[1].set_title("Rescaled image")

ax[2].imshow(image_resized, cmap='gray')
ax[2].set_title("Resized image")

ax[3].imshow(image_downscaled, cmap='gray')
ax[3].set_title("Downscaled image using Gaussian smoothing")

ax[4].imshow(image_downscaled_local_mean, cmap='gray')
ax[4].set_title("Downscaled image using local averaging")

ax[0].set_xlim(0, 512)
ax[0].set_ylim(512, 0)
plt.tight_layout()
plt.show()
