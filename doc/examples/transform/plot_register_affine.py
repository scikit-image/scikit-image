"""
=====================================
Affine Registration
=====================================

In this example, we use image registration to find an affine transformation
which can be used to align our target and reference images.

The :func:`skimage.measure.register_affine` function uses a Gaussian pyramid,
a cost function, and an optimization function to find the affine transformation
that best converts a reference space to a target space. This transformation
(which is expressed as an (ndim+1, ndim+1) matrix) can be used by
:func:`scipy.ndimage.affine_transform` to convert the target image to the
reference space.

.. [1] Juan Nunez-Iglesias, Stefan van der Walt, and Harriet Dashnow. Elegant
        SciPy: The Art of Scientific Python. 1st. O'Reilly Media, Inc.,
        2017. isbn: 1491922877, 9781491922873.

"""

import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt

from skimage.data import camera
from skimage.transform import register_affine, pyramid_gaussian
from skimage import measure


###############################################################################
# First, we make a toy example with an image and a shifted and rotated version
# of the same image, using a transformation matrix. See the Wikipedia page on
# `homogeneous coordinates`_ for information on this step.
#
# .. _homogeneous coordinates: https://en.wikipedia.org/wiki/Homogeneous_coordinates

r = 0.12
c, s = np.cos(r), np.sin(r)
matrix_transform = np.array([[c, -s, 0],
                             [s, c, 50],
                             [0, 0,  1]])

image = camera()
target = ndi.affine_transform(image, matrix_transform)

###############################################################################
# Next, we are going to see how ``register_affine`` can recover that
# transformation starting from only the two images. The registration works by
# nudging the input image slightly and checking whether it is closer or further
# away from the reference image. It does this initially on a much blurrier and
# smaller version of the two images, then progressively refines the alignment
# with sharper, full-resolution versions. This is called a Gaussian pyramid.
# ``register_affine`` also allows a *callback* function to be passed, which is
# executed at every level of the Gaussian pyramid. We can use the callback to
# observe the process of alignment.

level_alignments = []

register_matrix = register_affine(image, target,
                                  level_callback=level_alignments.append)

###############################################################################
# Once we have the matrix, it's easy to transform the target image to match
# the reference using :func:`scipy.ndimage.affine_transform`:

registered = ndi.affine_transform(target, register_matrix)

###############################################################################
# Let's look at the results. First, we make a helper function to overlay two
# grayscale images as yellow (reference) and cyan (target):


def overlay(image0, image1):
    """Overlay two grayscale images as yellow and cyan channels.

    The images must have the same shape.
    """
    zeros = np.zeros_like(image0)
    image0_color = np.stack((image0, image0, zeros), axis=-1)
    image1_color = np.stack((zeros, image1, image1), axis=-1)
    image_overlay = np.maximum(image0_color, image1_color)
    return image_overlay


###############################################################################
# Now we can look at the alignment. The reference image is in yellow, while the
# target image is in cyan. Regions of perfect overlap become gray or black:

_, ax = plt.subplots(1, 2)

ax[0].set_title('initial alignment')
ax[0].imshow(overlay(image, target))

ax[1].set_title('registered')
ax[1].imshow(overlay(image, registered))

for a in ax:
    a.set_axis_off()

plt.show()

###############################################################################
# Let's observe the Gaussian pyramid at work, as described above, using the
# per-level alignments that we saved to a list using ``level_callback``.

_, ax = plt.subplots(1, 6)

initial_nmi = measure.compare_nmi(image, target)
ax[0].set_title('starting NMI {:.3}'.format(initial_nmi))
ax[0].imshow(overlay(image, target))

final_nmi = measure.compare_nmi(image, registered)
ax[5].set_title('final correction, NMI {:.3}'.format(final_nmi))
ax[5].imshow(overlay(image, registered))

###############################################################################
# We make a small function to add a grid to a displayed image for easy
# reference to the alignment:


def add_grid(ax, image):
    r, c = image.shape
    ax.set_xticks(np.arange(c / 5, c, c / 5), minor=True)
    ax.set_yticks(np.arange(r / 5, r, r / 5), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

###############################################################################
# Finally, we show the intermediate alignments with blurred images to
# demonstrate how registration with Gaussian pyramids works. For illustrative
# purposes only, we have to recreate the Gaussian pyramid outside of the
# registration function.


def gaussian_sigma(level):
    """Compute the equivalent blur to a given Gaussian pyramid level.

    The blur at a single level is 2/3 * 2**level, because the sigma is 2/3
    but the image at each level has been downsampled by a factor of 2.

    The total blur at that level, though, includes all lower levels of blur.
    Blurring by consecutive Gaussian filters is equivalent to blurring by
    a single filter with sigma equal to the square root of the sum of squared
    individual sigmas.
    """
    return np.sqrt(sum((2/3 * 2**curr_level)**2
                       for curr_level in range(level)))


num_levels = len(level_alignments)
reference_pyramid = list(reversed(list(pyramid_gaussian(image, max_layer=num_levels-1))))
for level_num in range(2, 6):
    level = num_levels - level_num - 1  # 0 is original image
    iter_target, matrix, nnmi = level_alignments[level_num]
    transformed_full = ndi.affine_transform(target, matrix)
    transformed = ndi.affine_transform(iter_target, matrix)
    level_image = reference_pyramid[level_num]
    # NMI is sensitive to image resolution, so we must compare at top level
    level_nmi = measure.compare_nmi(image, transformed_full)
    ax[level_num-1].set_title('level {}, NMI {:.3}'.format(level, level_nmi))
    ax[level_num-1].imshow(overlay(level_image, transformed),
                           interpolation='bilinear')
    ax[level_num-1].set_axis_off()

plt.show()
