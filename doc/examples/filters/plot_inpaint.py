"""
===============================
Fill in defects with inpainting
===============================

Inpainting [1]_ is the process of reconstructing lost or deteriorated
parts of images and videos.

The reconstruction (restoration) is performed in an automatic way by
exploiting the information present in non-damaged regions.

In this example, we show how the masked pixels get inpainted using an
inpainting algorithm based on the biharmonic equation [2]_ [3]_ [4]_.

.. [1]  Wikipedia. Inpainting
        https://en.wikipedia.org/wiki/Inpainting
.. [2]  Wikipedia. Biharmonic equation
        https://en.wikipedia.org/wiki/Biharmonic_equation
.. [3]  S.B.Damelin and N.S.Hoang. "On Surface Completion and Image
        Inpainting by Biharmonic Functions: Numerical Aspects",
        International Journal of Mathematics and Mathematical Sciences,
        Vol. 2018, Article ID 3950312
        :DOI:`10.1155/2018/3950312`
.. [4]  C. K. Chui and H. N. Mhaskar, MRA Contextual-Recovery Extension of
        Smooth Functions on Manifolds, Appl. and Comp. Harmonic Anal.,
        28 (2010), 104-113,
        :DOI:`10.1016/j.acha.2009.04.004`
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology import disk, binary_dilation
from skimage.restoration import inpaint

image_orig = data.astronaut()

# Create mask with six block defect regions
mask = np.zeros(image_orig.shape[:-1], dtype=bool)
mask[20:60, 0:20] = 1
mask[160:180, 70:155] = 1
mask[30:60, 170:195] = 1
mask[-60:-30, 170:195] = 1
mask[-180:-160, 70:155] = 1
mask[-60:-20, 0:20] = 1

# Add a few long, narrow defects
mask[200:205, -200:] = 1
mask[150:255, 20:23] = 1
mask[365:368, 60:130] = 1

# Add randomly positioned small point-like defects
rstate = np.random.default_rng(0)
for radius in [0, 2, 4]:
    # larger defects are less common
    thresh = 3 + 0.25 * radius  # make larger defects less common
    tmp_mask = rstate.standard_normal(image_orig.shape[:-1]) > thresh
    if radius > 0:
        tmp_mask = binary_dilation(tmp_mask, disk(radius, dtype=bool))
    mask[tmp_mask] = 1

# Apply defect mask to the image over the same region in each color channel
image_defect = image_orig * ~mask[..., np.newaxis]

image_result = inpaint.inpaint_biharmonic(image_defect, mask, channel_axis=-1)

fig, axes = plt.subplots(ncols=2, nrows=2)
ax = axes.ravel()

ax[0].set_title('Original image')
ax[0].imshow(image_orig)

ax[1].set_title('Mask')
ax[1].imshow(mask, cmap=plt.cm.gray)

ax[2].set_title('Defected image')
ax[2].imshow(image_defect)

ax[3].set_title('Inpainted image')
ax[3].imshow(image_result)

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()
