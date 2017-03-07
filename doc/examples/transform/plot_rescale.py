"""
=======
Rescale
=======
The rescale operation resizes an image by a given scaling factor.
The scaling factor can either be a single floating point value,
or multiple values---one along each axis.
"""


import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import rescale

image = data.camera()

rescale_image = rescale(image, 0.5)

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                               sharex=True, sharey=True,
                               subplot_kw={'adjustable':'box-forced'})

ax0.imshow(image, cmap=plt.cm.gray)
ax0.axis('off')
ax0.set_title("Original Image")

ax1.imshow(rescale_image, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title("Rescaled Image")

plt.show()
