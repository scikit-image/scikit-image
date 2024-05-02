"""
=====================================
Affine Registration
=====================================

In this example, we use image registration to find an affine transformation
which can be used to align our target and reference images.

The :func:`skimage.measure.register_affine` function uses a Gaussian pyramid,
a cost function, and an optimization function to find the affine transformation
that best aligns a reference image to a target image. This transformation
(which is expressed as an (ndim+1, ndim+1) matrix) can be used by
:func:`scipy.ndimage.affine_transform` to convert the target image to the
reference space. This approach is explained in detail in Chapter 7 of Elegant
SciPy [1]_.

.. [1] Juan Nunez-Iglesias, Stefan van der Walt, and Harriet Dashnow. Elegant
        SciPy: The Art of Scientific Python. 1st. O'Reilly Media, Inc.,
        2017. isbn: 1491922877, 9781491922873.

"""

import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt

from skimage.data import astronaut
from skimage.transform import pyramid_gaussian
from skimage import registration
from skimage import metrics


###############################################################################
# First, we make a toy example with an image and a shifted and rotated version
# of the same image, using a transformation matrix. See the Wikipedia page on
# `homogeneous coordinates`_ for information on this step.
#
# .. _homogeneous coordinates: https://en.wikipedia.org/wiki/Homogeneous_coordinates

r = -0.12
c, s = np.cos(r), np.sin(r)
matrix_transform = np.array([[c, -s, 0],
                             [s, c, 50],
                             [0, 0,  1]])

image = astronaut()[..., 1]  # Just green channel
target = ndi.affine_transform(image, matrix_transform)

###############################################################################
# Next, we are going to see how ``registration.affine`` can recover that
# transformation starting from only the two images. The registration works by
# nudging the input image slightly and checking whether it is closer or further
# away from the reference image. It does this initially on a much blurrier and
# smaller version of the two images, then progressively refines the alignment
# with sharper, full-resolution versions. This is called a Gaussian pyramid.
# ``registration.affine`` also allows a *callback* function to be passed, which
# is executed at every level of the Gaussian pyramid. We can use the callback
# to observe the process of alignment.

level_alignments = []


import time
t0 = time.time()
register_matrix = registration.affine(image, target,
                                      level_callback=level_alignments.append)
t1 = time.time()

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
# per-level alignments that we saved to a list using ``level_callback``. We
# show the intermediate alignments with blurred images to demonstrate how
# registration with Gaussian pyramids works. For illustrative purposes only, we
# have to recreate the Gaussian pyramid outside of the registration function.

_, ax = plt.subplots(1, 6)

initial_nmi = metrics.normalized_mutual_information(image, target)
ax[0].set_title('starting NMI {:.3}'.format(initial_nmi))
ax[0].imshow(overlay(image, target))

final_nmi = metrics.normalized_mutual_information(image, registered)
ax[5].set_title('final correction, NMI {:.3}'.format(final_nmi))
ax[5].imshow(overlay(image, registered))

num_levels = len(level_alignments)

reference_pyramid = list(pyramid_gaussian(image, max_layer=num_levels-1))[::-1]
for axis_num, level_num in enumerate([0, 2, 4, 5], start=1):
    level = num_levels - level_num - 1  # 0 is original image
    iter_target, matrix, nnmi = level_alignments[level_num]
    transformed_full = ndi.affine_transform(target, matrix)
    transformed = ndi.affine_transform(iter_target, matrix)
    level_image = reference_pyramid[level_num]
    # NMI is sensitive to image resolution, so we must compare at top level
    level_nmi = metrics.normalized_mutual_information(image, transformed_full)
    ax[axis_num].set_title('level {}, NMI {:.4}'.format(level, level_nmi))
    ax[axis_num].imshow(overlay(level_image, transformed),
                           interpolation='bilinear')
    ax[axis_num].set_axis_off()

plt.show()

###############################################################################
# If we know that our transform is a *rigid* transform, also known as a
# Euclidean transform, we can use scikit-image's transformation classes to use
# a smaller parameter set over which to optimize. This can make the
# registration faster and more robust.
#
# It is important to note that out of the components of an affine
# transformation, scale, rotation, and skew are scale-invariant, but
# translation is not. Therefore, parameters representing translation in the
# parameter vector need to be rescaled between different levels of the
# pyramid. `registration.affine` does this automatically, but it needs to know
# which parts of the parameter vector represent translation. The keyword
# argument ``translation_indices`` is provided for this purpose.
#
# Finally, in this case, a too-small pyramid image causes the registration to
# fail to converge to the correct result, so we set the smallest size to 64.

from skimage.transform import EuclideanTransform


def rigid_transform(params):
    return EuclideanTransform(rotation=params[0], translation=params[1:])


t2 = time.time()
rigid_matrix = registration.affine(image, target,
                                   initial_parameters=np.zeros(3),
                                   vector_to_matrix=rigid_transform,
                                   pyramid_minimum_size=64,
                                   translation_indices=slice(1, None))
t3 = time.time()


print('original matrix:')
print(matrix_transform)
print(f'full registration result in {t1 - t0:.1f} seconds:')
print(register_matrix)
print(f'rigid registration result in {t3 - t2:.1f} seconds:')
print(rigid_matrix)
