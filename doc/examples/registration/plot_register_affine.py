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
from skimage import registration


###############################################################################
# First, we make a toy example with an image and a shifted and rotated version
# of the same image, using a transformation matrix. See the Wikipedia page on
# `homogeneous coordinates`_ for information on this step.
#
# .. _homogeneous coordinates: https://en.wikipedia.org/wiki/Homogeneous_coordinates


reference = astronaut()[..., 1]  # Just green channel

r = -0.12  # radians
c, s = np.cos(r), np.sin(r)
T = np.array(
    [[1, 0, -reference.shape[0] / 2], [0, 1, -reference.shape[1] / 2], [0, 0, 1]]
)
R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
matrix_transform = np.linalg.inv(T) @ R @ T
# matrix_transform = np.array([[c, -s, 0], [s, c, 50], [0, 0, 1]])

moving = ndi.affine_transform(reference, matrix_transform)

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
    # TODO: this is not nice
    if solver is registration.studholme_affine_solver:
        matrix = registration._affine._parameter_vector_to_matrix(matrix, 2)
    results.append(
        {
            "test": solver.__name__,
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


def target_registration_error(shape, matrix):
    """Compute the displacement norm for the transformation"""
    # Create a regular set of points on the grid
    points = np.concatenate(
        [
            np.stack(
                [
                    x.flatten()
                    for x in np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
                ],
                axis=1,
            ).T,
            np.array([1] * np.prod(shape)).reshape(1, -1),
        ]
    )
    delta = matrix @ matrix_transform @ points - points
    return np.linalg.norm(delta[: len(shape)], axis=0).reshape(shape)


###############################################################################
# Now we can look at the alignment. The reference image is in yellow, while the
# target image is in cyan. Regions of perfect overlap become gray or black:

_, ax = plt.subplots(2, len(results) + 1)

ax[0, 0].set_title("initial alignment")
ax[0, 0].imshow(overlay(reference, moving))

for k, item in enumerate(results):
    ax[0, k + 1].set_title(item["test"])
    ax[0, k + 1].imshow(overlay(reference, item["registered"]))
    ax[1, k + 1].set_title("TRE")
    ax[1, k + 1].imshow(
        target_registration_error(reference.shape, item["matrix"] @ matrix_transform)
    )

for a in ax.ravel():
    a.set_axis_off()

plt.show()


###############################################################################
# If we know that our transform is a *rigid* transform, also known as a
# Euclidean transform.
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

from functools import partial


def p2m_rigid(params, ndim):
    """Rigid transform homogenous matrix

    Note this respect the ij convention of ndi.affine_transform
    """
    c = np.cos(params[0])
    s = np.sin(params[0])
    return np.array([[c, s, params[1]], [-s, c, params[2]], [0, 0, 1]])


solver = partial(registration.studholme_affine_solver, vector_to_matrix=p2m_rigid)

start_time = time.time()
params = registration.affine(
    reference, moving, solver=solver, translation_indices=[1, 2], matrix=[0, 0, 0]
)
end_time = time.time()
results.append(
    {
        "test": "rigid motion",
        "elapsed time": end_time - start_time,
        "matrix": p2m_rigid(params, 2),
        "registered": ndi.affine_transform(moving, p2m_rigid(params, 2)),
    }
)

print("original matrix:")
print(np.linalg.inv(matrix_transform))
for item in results:
    item["tre"] = target_registration_error(
        reference.shape, item["matrix"] @ matrix_transform
    )
    print(
        f"""registration result with {item["test"]}
         in {item["elapsed time"]:.2f} seconds, TRE {item["tre"].mean():.2f}
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
    ax[1, k + 1].set_title("TRE")
    ax[1, k + 1].imshow(item["tre"])

for a in ax.ravel():
    a.set_axis_off()

plt.show()
