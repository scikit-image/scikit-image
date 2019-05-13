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
from skimage.transform import register_affine
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
# Looking at the results and the registration process:

_, axes = plt.subplots(2, 3)
ax = axes.ravel()

ax[0].set_title('reference')
ax[0].imshow(image, cmap='gray')

initial_nmi = measure.compare_nmi(image, target)
ax[1].set_title('target, NMI {:.3}'.format(initial_nmi))
ax[1].imshow(target, cmap='gray')

final_nmi = measure.compare_nmi(image, registered)
ax[2].set_title('final correction, NMI {:.3}'.format(final_nmi))
ax[2].imshow(registered, cmap='gray')

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

for i in range(3):
    add_grid(ax[i], image)

###############################################################################
# Finally, we show the intermediate alignments with blurred images to
# demonstrate how registration with Gaussian pyramids works:

for axis_num, level_num in enumerate([1, 2, 4], start=3):
    iter_target, matrix, nnmi = level_alignments[level_num]
    transformed_target = ndi.affine_transform(iter_target, matrix)
    # NMI is sensitive to image resolution, so must compare at top level
    level_nmi = measure.compare_nmi(image,
                                    ndi.affine_transform(target, matrix))
    ax[axis_num].set_title('level {}, nmi {:.3}'.format(level_num, level_nmi))
    ax[axis_num].imshow(transformed_target, cmap='gray',
                        interpolation='gaussian', resample=True)
    add_grid(ax[axis_num], transformed_target)

plt.show()
