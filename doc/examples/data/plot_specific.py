"""
===============
Specific images
===============

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data

# TODO
#'lfw_subset',

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

images = data.stereo_motorcycle()
ax[0].imshow(images[0])
ax[1].imshow(images[1])

#ax[i].set_title(image)

fig.tight_layout()
plt.show()
