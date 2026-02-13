"""
=================================================
Estimate affine transformation to register images
=================================================

In this example, we use image registration to find an affine transformation
which can be used to align a moving image to a reference image.

The :func:`skimage.measure.register_affine` function uses a Gaussian pyramid,
and a solver to estimate the parameter of an affine transformation model that
best aligns the moving image to a reference image. This transformation (which
is expressed as a (ndim+1, ndim+1) matrix) can be used by
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

import skimage as ski

###############################################################################
# First, we make a toy example with an image and a shifted and rotated version
# of the same image, using a transformation matrix. See the Wikipedia page on
# `homogeneous coordinates`_ for information on this step.
#
# .. _homogeneous coordinates: https://en.wikipedia.org/wiki/Homogeneous_coordinates


reference = ski.data.camera()

# Define a rotation around the center of the image
r = -0.12  # rotation angle in radians
c, s = np.cos(r), np.sin(r)
R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
# translation to center the rotation
T = np.array(
    [[1, 0, -reference.shape[0] / 2], [0, 1, -reference.shape[1] / 2], [0, 0, 1]]
)
matrix = np.linalg.inv(T) @ R @ T

moving = ndi.affine_transform(reference, matrix)

###############################################################################
# Next, we are going to see how ``ski.registration.estimate_affine`` can recover
# that transformation starting from only the two images. It does this initially on a
# much blurrier and smaller version of the two images, then progressively
# refines the alignment with sharper, full-resolution versions. This is called
# a Gaussian pyramid. This function can take different solvers for estimating
# the transformation at each pyramid level.
# Solvers are LucasKanadeAffineSolver, StudholmeAffineSolver and ECCAffineSolver
# Each solver take a model as first parameter which can be TranslationTransform,
# EuclideanTransform or AffineTransform

model_class = ski.registration.EuclideanTransform
solver = ski.registration.LucasKanadeAffineSolver(model_class)
transform = ski.registration.estimate_affine(reference, moving, solver=solver)

################################################################################
# To align the moving image, we use ``ndi.affine_transform``

registered = ndi.affine_transform(moving, transform.params)


###############################################################################
# Becauser we know the original transform, we can also compute the target
# registration error map:

tre = ski.registration.target_registration_error(
    reference.shape, transform.params @ matrix
)

###############################################################################
# Let's have a look at the results. We display here images pairs as
# magenta-green color images.

plt.subplot(131)
plt.imshow(np.stack((reference, moving, reference), -1))
plt.axis('off')
plt.title('Before registation')
plt.subplot(132)
plt.imshow(np.stack((reference, registered, reference), -1))
plt.axis('off')
plt.title('After registation')
plt.subplot(133)
im = plt.imshow(tre)
plt.axis('off')
plt.title('TRE')
plt.tight_layout()
plt.show()
