"""
=================================================
Estimate affine transformation to register images
=================================================

In this example, we compute an affine transformation
which may be used to align (register) a moving image to a reference image.

The :func:`skimage.measure.estimate_affine` function uses a Gaussian pyramid
and a solver to estimate the parameters of an affine transformation model that
best aligns (registers) the moving image to the reference image. This
transformation, which is expressed as a (ndim+1, ndim+1) matrix, may be used by
:func:`scipy.ndimage.affine_transform` to convert the moving image to the
reference space. This approach is explained in detail in Chapter 7 of *Elegant
SciPy* [1]_. Note that the moving image is denoted by ``target_image`` or
``target`` in [1]_.

.. [1] Juan Nunez-Iglesias, Stefan van der Walt, and Harriet Dashnow. Elegant
       SciPy: The Art of Scientific Python. 1st. O'Reilly Media, Inc.,
       2017. isbn: 1491922877, 9781491922873.

"""

import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt

import skimage as ski

###############################################################################
# First, we create a toy example with an image, and a shifted and rotated version
# of the same image, by using a transformation matrix. See the Wikipedia page on
# `homogeneous coordinates`_ for information on this step.
#
# .. _homogeneous coordinates: https://en.wikipedia.org/wiki/Homogeneous_coordinates


reference = ski.data.camera()

# rotation around the center of the image
r = -np.pi / 4  # rotation angle in radians
c, s = np.cos(r), np.sin(r)
R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
# translation to center the rotation
T = np.array(
    [[1, 0, -reference.shape[0] / 2], [0, 1, -reference.shape[1] / 2], [0, 0, 1]]
)
matrix = np.linalg.inv(T) @ R @ T

moving = ndi.affine_transform(reference, matrix)

###############################################################################
# Next, we are going to see how :func:`skimage.measure.estimate_affine` can recover
# that transformation starting from only the two images. It does this initially on a
# much blurrier and smaller version of the two images, then progressively
# refines the alignment with sharper, full-resolution versions. That is called
# a Gaussian pyramid. The function may use different kinds of solvers for estimating
# the transformation at each pyramid level. Available solvers are
# "lukas-kanade", "studholme", and "ecc".


transform = ski.registration.estimate_affine(
    reference, moving, transform_type="affine", solver_config="lukas-kanade"
)

###############################################################################
# To register the moving image, we use :func:`scipy.ndimage.affine_transform`:

registered = ndi.affine_transform(moving, transform.params)


###############################################################################
# Since we know the original transform, we can also compute the target
# registration error (TRE):

tre = ski.registration.target_registration_error(
    reference.shape, transform.params @ matrix
)

###############################################################################
# Let us have a look at the results. Below we display image pairs as
# magenta-green color images.

fig, ax = plt.subplots(ncols=3)

ax[0].imshow(np.stack((reference, moving, reference), -1))
ax[0].set_title('Before registration')
ax[0].axis('off')

ax[1].imshow(np.stack((reference, registered, reference), -1))
ax[1].set_title('After registration')
ax[1].axis('off')

ax[2].imshow(tre)
ax[2].set_title(f'TRE; max = {tre.max():.2g}')
ax[2].axis('off')

fig.tight_layout()

plt.show()
