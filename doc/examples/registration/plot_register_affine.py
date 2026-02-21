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


reference = ski.data.camera()[::4, ::4]

# Define a rotation around the center of the image
r = -0.12  # rotation angle in radians
c, s = np.cos(r), np.sin(r)
R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
# translation to center the rotation
T = np.array(
    [[1, 0, -reference.shape[0] / 2], [0, 1, -reference.shape[1] / 2], [0, 0, 1]]
)
transform = np.linalg.inv(T) @ R @ T
moving = ndi.affine_transform(reference, transform)

###############################################################################
# Next, we are going to see how ``ski.registration.affine`` can recover that
# transformation starting from only the two images. It does this initially on a
# much blurrier and smaller version of the two images, then progressively
# refines the alignment with sharper, full-resolution versions. This is called
# a Gaussian pyramid. This function can take two different solvers for estimating
# the transformation at each pyramid level.
# ``ski.registration.affine``


import time


solvers = [
    ski.registration.solver_affine_lucas_kanade,
    ski.registration.solver_affine_ecc,
    ski.registration.solver_affine_studholme,
]

results = []
for solver in solvers:
    start_time = time.time()
    matrix = ski.registration.affine(reference, moving, solver=solver)
    stop_time = time.time()
    results.append(
        {
            "test": "Affine / " + solver.__name__.replace("solver_affine_", ""),
            "elapsed time": stop_time - start_time,
            "matrix": matrix,
        }
    )


###############################################################################
# Once we have the matrix, it's easy to transform the target image to match
# the reference using :func:`scipy.ndimage.affine_transform`:

for item in results:
    item["registered"] = ndi.affine_transform(moving, item["matrix"])

###############################################################################
# We can also compute a registration error map:

for item in results:
    item["tre"] = ski.registration.target_registration_error(
        reference.shape, item["matrix"] @ transform
    )

###############################################################################
# Let's look at the results. First, we make a helper function to overlay two
# grayscale images as magenta (reference) and green (target):


def overlay(reference, moving):
    """Overlay two grayscale images as magenta and green channels.

    Parameters
    ----------
    reference: np.ndarray
        The reference image with shape (H,W).
    moving: np.ndarray
        The moving image with shape (H,W).

    Returns
    -------
    image_overlay: np.ndarray
        The RGB image as a (H,W,C) array.

    Note
    ----
    The images must have the same shape.
    """
    image_overlay = np.stack((reference, moving, reference), -1)
    return image_overlay


###############################################################################
# Now we can look at the alignment. The reference image is in magenta, while the
# target image is in green. Regions of perfect overlap become white

_, ax = plt.subplots(2, len(results) + 1)

ax[0, 0].set_title("initial alignment")
ax[0, 0].imshow(overlay(reference, moving))
tre_initial = ski.registration.target_registration_error(reference.shape, transform)
ax[1, 0].set_title(f"TRE (max:{tre_initial.max():.2f}px)")
ax[1, 0].imshow(tre_initial)

for k, item in enumerate(results):
    ax[0, k + 1].set_title(item["test"])
    ax[0, k + 1].imshow(overlay(reference, item["registered"]))
    ax[1, k + 1].set_title(f"TRE (max:{item['tre'].max():.2f}px)")
    ax[1, k + 1].imshow(item["tre"])


for a in ax.ravel():
    a.set_axis_off()

plt.show()


###############################################################################
# If we know that our transform is a *rigid* transform, also known as a
# Euclidean transform, we can reduce the number of free parameters in the model.
#
for solver in solvers:
    start_time = time.time()
    matrix = ski.registration.affine(
        reference, moving, solver=solver, model="euclidean"
    )
    stop_time = time.time()
    results.append(
        {
            "test": "Euclidean / " + solver.__name__.replace("solver_affine_", ""),
            "elapsed time": stop_time - start_time,
            "matrix": matrix,
            "registered": ndi.affine_transform(moving, matrix),
            "tre": ski.registration.target_registration_error(
                reference.shape, matrix @ transform
            ),
        }
    )


print("Original matrix:")
print(np.linalg.inv(transform))

for item in results:
    print(
        f"{item['test']} in {item['elapsed time']:.2f} seconds,"
        f" TRE max:{item['tre'].max():.2f}/mean:{item['tre'].mean():.2f} pixels."
    )
    print(item["matrix"])

###############################################################################
# Now we can look at the alignment. The reference image is in magenta, while the
# target image is in green. Regions of perfect overlap have are in grayscale:

_, ax = plt.subplots(2, len(results) + 1)

ax[0, 0].set_title("Initial alignment")
ax[0, 0].imshow(overlay(reference, moving))
ax[1, 0].set_title("Initial alignment")
ax[1, 0].set_title(f"TRE (max:{tre_initial.max():.2f}px)")
ax[1, 0].imshow(tre_initial)

for k, item in enumerate(results):
    ax[0, k + 1].set_title(item["test"])
    ax[0, k + 1].imshow(overlay(reference, item["registered"]))
    ax[1, k + 1].set_title(f"TRE (max:{item['tre'].max():.2f}px)")
    ax[1, k + 1].imshow(item["tre"])

for a in ax.ravel():
    a.set_axis_off()

plt.show()
