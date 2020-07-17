"""
=======================
Background Subtraction using the Rolling Ball Method
=======================

The rolling ball filter is a segmentation method that aims to separate the
background from a grayscale image in case of uneven exposure. It is frequently
used in biomedical image processing and was first proposed by Stanley R.
Sternberg (1983) in the paper
[Biomedical Image Processing](https://ieeexplore.ieee.org/document/1654163).

The idea of the algorithm is quite intuitive. We think of the image as a
surface that has unit sized blockes stacked on top of each other for each
pixel. The number of blocks is determined by the intensity of a pixel. To get
the intensity of the background at a desired position, we imagine submerging a
ball into the blocks at the desired pixel position. Once it is completely
covered by the blocks, the height of the ball determines the intensity of the
background at that position. We can then *roll* this ball around below the
surface to get the the background values for the entire image. As Sternberg
puts it: "We can visualize a solid sphere that moves freely within the solid
volume of the gel image umbra but is constrained by the upper surface of the
umbra."

To implement this algorithm in skimage we can use the following:
"""

import numpy as np
from skimage import morphology


def rolling_ball(image, radius=30, white_background=False):
    working_img = image.copy()
    if white_background:
        working_img = 255 - working_img

    # tensor representation of the image
    # (stacked blocks as described in the paper)
    background = (np.arange(255)[np.newaxis,
                                 np.newaxis, :] < working_img[..., np.newaxis])

    # roll the ball around
    # here: implemented as a sequence of small balls (B1, ..., Bk)
    # because the current implementation of a closure with a large ball
    # is slow and consumes a lot of memory
    selem = morphology.ball(1)
    for idx in range(radius):
        background = morphology.erosion(background, selem)

    for idx in range(radius):
        background = morphology.dilation(background, selem)

    # flatten the tensor
    background = np.argmax(background == False, axis=-1)

    filtered_image = working_img - background

    if white_background:
        filtered_image = 255 - filtered_image
        background = 255 - background

    return filtered_image, background


######################################################################
# And then test it using an image with white background

from skimage import data
import matplotlib.pyplot as plt


image = data.page()

filtered_image, background = rolling_ball(image, radius=30,
                                          white_background=True)
fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original image')

ax[1].imshow(background, cmap='gray')
ax[1].set_title('Background')

ax[2].imshow(filtered_image, cmap='gray')
ax[2].set_title('Result')

plt.show()

######################################################################
# And with an image with black background

image = data.coins()

filtered_image, background = rolling_ball(image, radius=30,
                                          white_background=True)
fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original image')

ax[1].imshow(background, cmap='gray')
ax[1].set_title('Background')

ax[2].imshow(filtered_image, cmap='gray')
ax[2].set_title('Result')

plt.show()

######################################################################
# As you may have noticed, above implementation is rather slow.
# If runtime is a concern, we can implement an efficient approximation
# using a tophat filter and a disk element

from skimage import morphology


def rolling_ball(image, radius=50, white_background=False):
    selem = morphology.disk(radius)

    background = morphology.opening(image, selem)

    if white_background:
        filtered_image = morphology.black_tophat(image, selem)
    else:
        filtered_image = morphology.white_tophat(image, selem)

    return filtered_image, background

######################################################################
# Which produces almost identical results to the original algorithm
# albeit being much faster.

from skimage import data
import matplotlib.pyplot as plt


image = data.coins()

filtered_image, background = rolling_ball(image, radius=50)
fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original image')

ax[1].imshow(background, cmap='gray')
ax[1].set_title('Background')

ax[2].imshow(filtered_image, cmap='gray')
ax[2].set_title('Result')

plt.show()
