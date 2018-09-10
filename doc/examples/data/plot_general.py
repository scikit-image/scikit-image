"""
======================
General-purpose images
======================

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data


images = (
           'astronaut',
           'binary_blobs',
           'camera',
           'checkerboard',
           'chelsea',
           'clock',
           'coffee',
           'coins',
           'horse',
           'logo',
           'page',
           'text',
           'rocket',
           )

fig, axes = plt.subplots(len(images), 1, figsize=(8, 4 * len(images)))
ax = axes.ravel()

for i, image in enumerate(images):
    caller = getattr(data, image)
    ax[i].imshow(caller())
    ax[i].set_title(image)

fig.tight_layout()
plt.show()
