"""
==============
Frangi filter
==============

Frangi and hybrid Hessian filters can be used for edge detection and
calculation of part of the image containing edges.
"""

from skimage.data import camera
from skimage.filters import frangi, hessian

import matplotlib.pyplot as plt


image = camera()

fig, ax = plt.subplots(ncols=2, subplot_kw={'adjustable':'box-forced'})

ax[0].imshow(frangi(image), cmap=plt.cm.gray)
ax[0].set_title('Frangi filter results')

ax[1].imshow(hessian(image), cmap=plt.cm.gray)
ax[1].set_title('Hybrid Hessian filter result')

for a in ax:
    a.axis('off')

plt.tight_layout()
