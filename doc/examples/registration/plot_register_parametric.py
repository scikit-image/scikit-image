import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt

from skimage.data import astronaut, cells3d
from skimage import registration

import tifffile

import time

###############################################################################
# First, we make a toy example with an image and a shifted and rotated version
# of the same image, using a transformation matrix. See the Wikipedia page on
# `homogeneous coordinates`_ for information on this step.
#
# .. _homogeneous coordinates: https://en.wikipedia.org/wiki/Homogeneous_coordinates

image = astronaut()[..., 0]  # Just green channel

r = np.random.uniform(-0.2, 0.2)  # radians
c, s = np.cos(r), np.sin(r)
T = np.array([[1, 0, -image.shape[0] / 2], [0, 1, -image.shape[1] / 2], [0, 0, 1]])
R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
matrix_transform = np.linalg.inv(T) @ R @ T

target = ndi.affine_transform(image, matrix_transform)

###############################################################################
# Next, we are going to see how ``registration.affine`` can recover that
# transformation starting from only the two images.

t0 = time.time()
register_matrix = registration.affine(image, target)
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


def mse(image0, image1):
    """Mean square error where image0 and image 1 are not 0"""
    mask = registered > 0
    delta = image0 - image1
    return np.mean(delta[mask] ** 2)


###############################################################################
# Now we can look at the alignment. The reference image is in yellow, while the
# target image is in cyan. Regions of perfect overlap become gray or black:

_, ax = plt.subplots(1, 2)

ax[0].set_title(f"initial alignment {mse(image, target):.2f}")
ax[0].imshow(overlay(image, target))


ax[1].set_title(f"registered {mse(image, registered):.2f}")
ax[1].imshow(overlay(image, registered))

for a in ax:
    a.set_axis_off()

print("Lucas Kanade's method")
print(f"Elapsed time: {t1-t0:.2f} seconds.")
print(f"MSE: {mse(image, registered):.2f}.")

plt.show()

###############################################################################
# Next, we are going to see how ``registration.affine`` can recover that
# transformation starting from only the two images.

t0 = time.time()

register_matrix = registration.affine(
    image, target, solver=registration.studholme_affine_solver
)
t1 = time.time()

###############################################################################
# Once we have the matrix, it's easy to transform the target image to match
# the reference using :func:`scipy.ndimage.affine_transform`:

registered = ndi.affine_transform(target, register_matrix)

###############################################################################
# Now we can look at the alignment. The reference image is in yellow, while the
# target image is in cyan. Regions of perfect overlap become gray or black:

_, ax = plt.subplots(1, 2)

ax[0].set_title(f"initial alignment {mse(image, target):.2f}")
ax[0].imshow(overlay(image, target))

ax[1].set_title(f"registered {mse(image, registered):.2f}")
ax[1].imshow(overlay(image, registered))

for a in ax:
    a.set_axis_off()

print("Studholme's method")
print(f"Elapsed time: {t1-t0:.2f} seconds.")
print(f"MSE: {mse(image, registered):.2f}.")

plt.show()

###############################################################################
# Registration of a 3D volume
#

try:
    import pooch

    reference = tifffile.imread(
        sorted(
            pooch.retrieve(
                "https://graphics.stanford.edu/data/voldata/mrbrain-8bit.tar.gz",
                known_hash=None,
                processor=pooch.Untar(),
            )
        )
    )[::2, ::4, ::4].astype(np.float32)
except ModuleNotFoundError:
    print("Need pooch to download the 3D brain example dataset. Using 3D cells instead")
    reference = cells3d()[:, 0, ::4, ::4]

T = np.concatenate(
    [
        np.concatenate(
            [np.eye(3), -np.array(reference.shape).reshape(3, 1) / 2], axis=1
        ),
        [[0, 0, 0, 1]],
    ]
)
r1 = np.random.uniform(-0.2, 0.2)  # radians
c1, s1 = np.cos(r1), np.sin(r1)
R1 = np.array([[c1, -s1, 0, 0], [s1, c1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
r2 = np.random.uniform(-0.2, 0.2)  # radians
c2, s2 = np.cos(r2), np.sin(r2)
R2 = np.array([[c2, -s2, 0, 0], [s2, c2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
matrix_transform = np.linalg.inv(T) @ R2 @ R1 @ T
moving = ndi.affine_transform(reference, matrix_transform)
matrix = registration.affine(reference, moving)
registered = ndi.affine_transform(moving, matrix)
p = reference.shape[0] // 2
plt.subplot(131)
plt.imshow(reference[p])
plt.subplot(132)
plt.imshow(moving[p])
plt.subplot(133)
plt.imshow(registered[p])
plt.show()
