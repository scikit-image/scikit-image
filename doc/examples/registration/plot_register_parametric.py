import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt

from skimage.data import astronaut
from skimage import registration

import time


# reference_image = astronaut()[..., 0]
# r = -0.12  # radians
# c, s = np.cos(r), np.sin(r)
# matrix_transform = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# # matrix_transform = np.eye(2,3) + np.random.normal(0,0.1,(2,3))
# moving_image = ndi.affine_transform(reference_image, matrix_transform)

# matrix1 = registration.parametric_ilk(
#     reference_image,
#     moving_image,
#     num_warp=50,
#     tol=0.001,
#     pyramid_downscale=2,
#     pyramid_minimum_size=32,
# )
# registered1 = ndi.affine_transform(moving_image, matrix1)
# mse1 = np.sum((registered1 > 0) * (registered1 - reference_image) ** 2) / np.sum(
#     (registered1 > 0)
# )
# print(mse1)


# plt.subplot(121)
# plt.imshow(reference_image)
# plt.subplot(122)
# plt.imshow(moving_image)
# plt.subplot(123)
# plt.imshow(registered1)


###############################################################################
# First, we make a toy example with an image and a shifted and rotated version
# of the same image, using a transformation matrix. See the Wikipedia page on
# `homogeneous coordinates`_ for information on this step.
#
# .. _homogeneous coordinates: https://en.wikipedia.org/wiki/Homogeneous_coordinates

r = -0.12
c, s = np.cos(r), np.sin(r)
matrix_transform = np.array([[c, -s, 0], [s, c, 50], [0, 0, 1]])

image = astronaut()[..., 1]  # Just green channel
target = ndi.affine_transform(image, matrix_transform)

###############################################################################
# Next, we are going to see how ``registration.affine`` can recover that
# transformation starting from only the two images.

t0 = time.time()
register_matrix = registration.affine(image, target, method="lucas kanade")
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
register_matrix = registration.affine(image, target, method="studholme")
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
