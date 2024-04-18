r"""
==========================================
Interpolate images with thin-plate splines
==========================================

A conventional technique for interpolating surfaces over a set of data points
are thin-plate splines (TPS) [1]_ [2]_.
In an image context, given pairs of source and target control points, TPS can
be used to transform a space, which, in our case, is a 2D image.


.. [1] Wikipedia, Thin plate spline
       https://en.wikipedia.org/wiki/Thin_plate_spline

.. [2] Bookstein, Fred L. "Principal warps: Thin-plate splines and the
       decomposition of deformations." IEEE Transactions on pattern analysis and
       machine intelligence 11.6 (1989): 567–585.
       :DOI:`10.1109/34.24792`
       https://user.engineering.uiowa.edu/~aip/papers/bookstein-89.pdf


Deform an image
===============

Image deformation implies displacing the pixels of an image relative to one another.
In this example, we deform the (2D) image of an astronaut by using thin-plate splines.
In our image, we define 6 source and target points labeled "1-6": "1-4" are found near
the image corners, "5" near the left smile corner, and "6" in the right eye.
At the "1-4" points, there is no displacement.
Point "5" is displaced upward and point "6" downward.

We use TPS as a very handy interpolator for image deformation.
"""

import matplotlib.pyplot as plt
import numpy as np

import skimage as ski

astronaut = ski.data.astronaut()

src = np.array([[50, 50], [400, 50], [50, 400], [400, 400], [240, 150], [200, 100]])
dst = np.array([[50, 50], [400, 50], [50, 400], [400, 400], [276, 100], [230, 100]])

# Fit the thin-plate spline from source (src) to target (dst) points

warped_img = ski.future.tps_warp(astronaut, src[:, ::-1], dst[:, ::-1], grid_scaling=1)


fig, axs = plt.subplots(1, 2)

labels = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
]  # Adjust the number of labels to match the number of points

axs[0].imshow(astronaut, cmap='gray')
axs[0].scatter(src[:, 0], src[:, 1], marker='x', color='cyan')

for i, label in enumerate(labels):
    axs[0].annotate(
        label,
        (src[:, 0][i], src[:, 1][i]),
        textcoords="offset points",
        xytext=(0, 5),
        ha='center',
        color='red',
    )

axs[1].imshow(warped_img, cmap='gray')
axs[1].scatter(dst[:, 0], dst[:, 1], marker='x', color='cyan')

for i, label in enumerate(labels):
    axs[1].annotate(
        label,
        (dst[:, 0][i], dst[:, 1][i]),
        textcoords="offset points",
        xytext=(0, 5),
        ha='center',
        color='red',
    )

plt.show()

######################################################################
#
# Derive an interpolation function
# ================================
# In this second example, we start with a set of source and target coordinates.
# TPS is applied to each source and target coordinate to derive an interpolation
# function and coefficients.
# These coefficients are then used to transform an arbitrary point associated
# with the reference to an interpolated location on the target.

import matplotlib.pyplot as plt

import skimage as ski

samp = np.linspace(-2, 2, 4)
xx, yy = np.meshgrid(samp, samp)

# Create source points
source_xy = np.column_stack((xx.ravel(), yy.ravel()))

# Create target points
yy[:, [0, 3]] *= 2
target_xy = np.column_stack((xx.ravel(), yy.ravel()))


# Compute the coefficient
trans = ski.future.TpsTransform()
trans.estimate(source_xy, target_xy)


# Create another arbitrary point
samp2 = np.linspace(-1.8, 1.8, 10)
test_xy = np.tile(samp2, [2, 1]).T

# Estimate transformed points from given sets of source and target points
transformed_xy = trans(test_xy)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 3))

ax0.scatter(source_xy[:, 0], source_xy[:, 1], label='Source Points')
ax0.scatter(test_xy[:, 0], test_xy[:, 1], c='orange', label='Test Points')
ax0.legend(loc='upper center')
ax0.set_title('Source and Test Coordinates')

ax1.scatter(target_xy[:, 0], target_xy[:, 1], label='Target Points')
ax1.scatter(
    transformed_xy[:, 0],
    transformed_xy[:, 1],
    c='orange',
    label='Transformed Test Points',
)
ax1.legend(loc="upper center")
ax1.set_title("Target and Transformed Coordinates")
plt.show()
