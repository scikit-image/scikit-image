"""
================================================================
Use the Rolling Ball Method for Background Intensity Estimation
================================================================

The rolling ball algorithm estimates background intensity of a grayscale
image in case of uneven exposure. It is frequently used in biomedical
image processing and was first proposed by Stanley R. Sternberg (1983) in
the paper Biomedical Image Processing [1]_.

The idea of the algorithm is quite intuitive. We think of the image as a
surface that has unit-sized blocks stacked on top of each other for each
pixel. The number of blocks is determined by the intensity of a pixel. To get
the intensity of the background at a desired position, we imagine submerging a
ball into the blocks at the desired pixel position. Once it is completely
covered by the blocks, the height of the ball determines the intensity of the
background at that position. We can then *roll* this ball around below the
surface to get the background values for the entire image.

Scikit-image gives you convenient access to this algorithm, and also implements
a generalized version which allows you to "roll" arbitrary ellipsoids. The
general version is useful when you want to use a different values for the
radius of the filter (``kernel_size``) and the amount (``intensity_vertex``).

.. [1] Sternberg, Stanley R. "Biomedical image processing." Computer 1 (1983):
    22-34. :DOI:`10.1109/MC.1983.1654163`


The Classic Rolling Ball Method
-------------------------------

In scikit-image, the rolling ball algorithm assumes that your background has
low intensity (black), whereas the features have high intensity (white). If
this is the case for your image, you can directly use the filter like:

"""

import matplotlib.pyplot as plt
import numpy as np
import time

from skimage import morphology
from skimage import data
from skimage import util


def plot_result(image, background):
    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')

    ax[1].imshow(background, cmap='gray')
    ax[1].set_title('Background')
    ax[1].axis('off')

    ax[2].imshow(image - background, cmap='gray')
    ax[2].set_title('Result')
    ax[2].axis('off')

    fig.tight_layout()


image = data.coins()

background = morphology.rolling_ball(image, radius=70.5)

plot_result(image, background)
plt.show()

######################################################################
# White background
# ----------------
#
# If you have dark features on a bright background, you need to invert
# the image before you pass it into the function, and then invert the
# result. This can be accomplished via:

image = data.page()
image_inverted = util.invert(image)

background_inverted = morphology.rolling_ball(image_inverted, radius=45)
filtered_image_inverted = image_inverted - background_inverted
filtered_image = util.invert(filtered_image_inverted)
background = util.invert(background_inverted)

fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original image')
ax[0].axis('off')

ax[1].imshow(background, cmap='gray')
ax[1].set_title('Background')
ax[1].axis('off')

ax[2].imshow(filtered_image, cmap='gray')
ax[2].set_title('Result')
ax[2].axis('off')

fig.tight_layout()

plt.show()

######################################################################
# Take care to not fall victim to an integer underflow when subtracting
# a bright background. For example, this code looks correct, but may
# suffer from an underflow leading to unwanted artifacts. You can see
# this in the top right corner of the visualization.

image = data.page()
image_inverted = util.invert(image)

background_inverted = morphology.rolling_ball(image_inverted, radius=45)
background = util.invert(background_inverted)
underflow_image = image - background  # integer underflow occurs here
correct_image = util.invert(background - image)  # correct subtraction

fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].imshow(underflow_image, cmap='gray')
ax[0].set_title('Background Removal with Underflow')
ax[0].axis('off')

ax[1].imshow(correct_image, cmap='gray')
ax[1].set_title('Correct Background Removal')
ax[1].axis('off')

fig.tight_layout()

plt.show()

######################################################################
# Image Datatypes
# ---------------
#
# ``rolling_ball`` can handle datatypes other than `np.uint8`. You can
# pass them into the function in the same way.

image = data.coins().astype(np.uint16)

background = morphology.rolling_ball(image, radius=70.5)
plot_result(image, background)
plt.show()

######################################################################
# However, you need to be careful if you use floating point images
# that have been normalized to ``[0, 1]``. In this case the ball will
# be much larger than the image intensity, which can lead to
# unexpected results.

image = util.img_as_float(data.coins())

background = morphology.rolling_ball(image, radius=70.5)
plot_result(image, background)
plt.show()

######################################################################
# Because ``radius=70.5`` is much larger than the maximum intensity of
# the image, the effective kernel size is reduced significantly, i.e.,
# only a small cap (approximately ``radius=10``) of the ball is rolled
# around in the image. You can find a reproduction of this strange
# effect in the ``rolling_ellipsoid`` section below.
#
# To get the expected result, you need to use the more flexible
# ``rolling_ellipsoid`` function. It works just like ``rolling_ball``;
# however, it gives you more control over the affected area and
# strength of the algorithm. In particular, it has different
# parameters for the spatial dimensions and the intensity dimension of
# the image.

image = util.img_as_float(data.coins())

normalized_radius = 70.5 / 255
background = morphology.rolling_ellipsoid(
    image,
    kernel_size=(70.5 * 2, 70.5 * 2),
    intensity_vertex=normalized_radius * 2
)
plot_result(image, background)
plt.show()

######################################################################
# Rolling Ellipsoid
# -----------------
#
# In ``rolling_ellipsoid`` you are specifying an ellipsoid instead of
# a ball/sphere - sidenote: a ball is a special case of an ellipsoid
# where each vertex has the same length. To fully specify an ellipsoid
# in 3D, you need to supply three parameters. Two for the two spacial
# dimensions of the image (via ``kernel_size``), and one for the
# intensity dimension (via ``intensity_vertex``).
#
# As mentioned above, a sphere is just a special ellipsoid, and hence
# you can get the same behavior as ``rolling_ball`` if you set all
# vertices to the respective values. In fact, ``rolling_ball``
# internally calls ``rolling_spheroid`` in the way shown below.
#
# Note: The radius is equal to the length of a semi-vertex of a
# sphere, which is *half* a full vertex. Hence, you need to multiply
# the inputs by two if you want to get the same result as
# ``rolling_ball``.

image = data.coins()

background = morphology.rolling_ellipsoid(
    image,
    kernel_size=(70.5 * 2, 70.5 * 2),
    intensity_vertex=70.5 * 2
)
plot_result(image, background)
plt.show()

######################################################################
# You can also use ``rolling_ellipsoid`` to recreate the previous,
# unexpected result and see that the effective (spacial) filter size
# was reduced.

image = data.coins()

background = morphology.rolling_ellipsoid(
    image,
    kernel_size=(10 * 2, 10 * 2),
    intensity_vertex=255 * 2
)
plot_result(image, background)
plt.show()
