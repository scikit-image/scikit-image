"""
=============
Frangi filter
=============

The Frangi and hybrid Hessian filters can be used to detect continuous
edges, such as vessels, wrinkles, and rivers.
"""

from skimage.data import camera
from skimage.filters import frangi, hessian

import matplotlib.pyplot as plt

image = camera()

fig, ax = plt.subplots(ncols=3, subplot_kw={'adjustable': 'box-forced'})

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(frangi(image), cmap=plt.cm.gray)
ax[1].set_title('Frangi filter result')

ax[2].imshow(hessian(image), cmap=plt.cm.gray)
ax[2].set_title('Hybrid Hessian filter result')

for a in ax:
    a.axis('off')

plt.tight_layout()
