"""
======================================================
Background Subtraction using the Rolling Ball Method
======================================================

The rolling ball filter is a segmentation method that aims to separate the
background from a grayscale image in case of uneven exposure. It is frequently
used in biomedical image processing and was first proposed by Stanley R.
Sternberg (1983) in the paper Biomedical Image Processing [1]_.

The idea of the algorithm is quite intuitive. We think of the image as a
surface that has unit-sized blocks stacked on top of each other for each
pixel. The number of blocks is determined by the intensity of a pixel. To get
the intensity of the background at a desired position, we imagine submerging a
ball into the blocks at the desired pixel position. Once it is completely
covered by the blocks, the height of the ball determines the intensity of the
background at that position. We can then *roll* this ball around below the
surface to get the background values for the entire image. As Sternberg
puts it: "We can visualize a solid sphere that moves freely within the solid
volume of the gel image umbra but is constrained by the upper surface of the
umbra."

.. [1] Sternberg, Stanley R. "Biomedical image processing." Computer 1 (1983):
    22-34. :DOI:`10.1109/MC.1983.1654163`

Use it with an image containing dark features on a black background
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import morphology
from skimage import data
from skimage.util import invert


image = data.page()

filtered_image, background = morphology.rolling_ball(image, radius=30,
                                                     white_background=True)
fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original image')

ax[1].imshow(background, cmap='gray')
ax[1].set_title('Background')

ax[2].imshow(filtered_image, cmap='gray')
ax[2].set_title('Result')

fig.tight_layout()
plt.show()

######################################################################
# or with an image containing white features on a black background

image = data.coins()

filtered_image, background = morphology.rolling_ball(image, radius=50)
fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original image')

ax[1].imshow(background, cmap='gray')
ax[1].set_title('Background')

ax[2].imshow(filtered_image, cmap='gray')
ax[2].set_title('Result')

fig.tight_layout()
plt.show()

######################################################################
# As you may have noticed, the above implementation is rather slow.
# If runtime is a concern, we can implement an efficient approximation
# using a tophat filter and a disk element


def rolling_disk(image, radius=50, white_background=False):
    selem = morphology.disk(radius)

    background = morphology.opening(image, selem)

    if white_background:
        filtered_image = morphology.black_tophat(image, selem)
    else:
        filtered_image = morphology.white_tophat(image, selem)

    return filtered_image, background

######################################################################
# This produces almost identical results to the original algorithm
# but is much faster.


image = data.coins()

filtered_image, background = rolling_disk(image, radius=50)
fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original image')

ax[1].imshow(background, cmap='gray')
ax[1].set_title('Background')

ax[2].imshow(filtered_image, cmap='gray')
ax[2].set_title('Result')

fig.tight_layout()
plt.show()
