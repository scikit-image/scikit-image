"""
===========================
Unsharp mask
===========================

In this example, we enhance the edge and detail information in an image using
a new method for unsharp masking. The algorithm employs an adaptive filter
that controls the contribution of the sharpening path in such a way that
contrast enhancement occurs in high detail areas and little or no image
sharpening occurs in smooth areas [1]_.

.. [1] Andrea Polesel, Giovanni Ramponi, and V. John Mathews (2000) "Image
       Enhancement via Adaptive Unsharp Masking" IEEE Trans. on Image
       Processing, 9(3): 505 - 510. DOI:10.1109/83.826787.
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import unsharp_mask


img = data.coins()
result = unsharp_mask(img)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                         sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(result, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_title('Enhanced image')

plt.tight_layout()
plt.show()
