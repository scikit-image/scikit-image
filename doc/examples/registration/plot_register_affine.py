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

from skimage.data import camera
from skimage import registration


###############################################################################
# First, we make a toy example with an image and a shifted and rotated version
# of the same image, using a transformation matrix. See the Wikipedia page on
# `homogeneous coordinates`_ for information on this step.
#
# .. _homogeneous coordinates: https://en.wikipedia.org/wiki/Homogeneous_coordinates


reference = camera()

r = -0.12  # radians
c, s = np.cos(r), np.sin(r)
T = np.array(
    [[1, 0, -reference.shape[0] / 2], [0, 1, -reference.shape[1] / 2], [0, 0, 1]]
)
R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
transform = np.linalg.inv(T) @ R @ T

moving = ndi.affine_transform(reference, transform)

###############################################################################
# Next, we are going to see how ``registration.affine`` can recover that
# transformation starting from only the two images. It does this initially on a
# much blurrier and smaller version of the two images, then progressively
# refines the alignment with sharper, full-resolution versions. This is called
# a Gaussian pyramid. This function can take two different solvers for estimating
# the transformation at each pyramid level.
# ``registration.affine``


import time

solvers = [
    registration.lucas_kanade_affine_solver,
    registration.studholme_affine_solver,
]

results = []
for solver in solvers:
    start_time = time.time()
    matrix = registration.affine(reference, moving, solver=solver)
    stop_time = time.time()
    results.append(
        {
            "test": solver.__name__.replace("_solver", ""),
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
    item["tre"] = registration.target_registration_error(
        reference.shape, item["matrix"] @ transform
    )

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

_, ax = plt.subplots(2, len(results) + 1)

ax[0, 0].set_title("initial alignment")
ax[0, 0].imshow(overlay(reference, moving))

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
    matrix = registration.affine(reference, moving, solver=solver, model="euclidean")
    stop_time = time.time()
    results.append(
        {
            "test": "euclidean"
            + solver.__name__.replace("_solver", "").replace("affine", "euclidean"),
            "elapsed time": stop_time - start_time,
            "matrix": matrix,
            "registered": ndi.affine_transform(moving, matrix),
            "tre": registration.target_registration_error(
                reference.shape, matrix @ transform
            ),
        }
    )


print("original matrix:")
print(np.linalg.inv(transform))
for item in results:
    print(
        f"""registration result with {item["test"]}
         in {item["elapsed time"]:.2f} seconds, TRE {item["tre"].max():.2f}
         """
    )
    print(item["matrix"])

###############################################################################
# Now we can look at the alignment. The reference image is in yellow, while the
# target image is in cyan. Regions of perfect overlap become gray or black:

_, ax = plt.subplots(2, len(results) + 1)

ax[0, 0].set_title("initial alignment")
ax[0, 0].imshow(overlay(reference, moving))

for k, item in enumerate(results):
    ax[0, k + 1].set_title(item["test"])
    ax[0, k + 1].imshow(overlay(reference, item["registered"]))
    ax[1, k + 1].set_title(f"TRE (max:{item['tre'].max():.2f}px)")
    ax[1, k + 1].imshow(item["tre"])

for a in ax.ravel():
    a.set_axis_off()

plt.show()
