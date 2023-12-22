r"""
==========================================
Interpolate images with thin-plate splines
==========================================

Thin-plate splines (TPS) refer to a method for interpolating data [1]_.
In an image context, we use a 2D generalization of TPS, i.e., a two-dimensional
simulation of a 1D cubic spline, which is the solution of the biharmonic equation [2]_.
According to [3]_, the basic solution of the biharmonic function was expanded.
Given pairs of source and target control points, TPS is used to transform a space,
which, in our case, is a 2D image.

This script shows how to use the Thine Plate Spine Transformation for
image warping.

For further information on TPS Transformation, see:

.. [1] Wikipedia, Thin plate spline
       https://en.wikipedia.org/wiki/Thin_plate_spline

.. [2] Weisstein, Eric W. "Biharmonic Equation." From MathWorld--A Wolfram Web Resource
       https://mathworld.wolfram.com/BiharmonicEquation.html

.. [3] Bookstein, Fred L. "Principal warps: Thin-plate splines and the
       decomposition of deformations." IEEE Transactions on pattern analysis and
       machine intelligence 11.6 (1989): 567–585.


Image Deformation
=================

In this example, we use the TPS interpolation to deform an image.
Deforming an image implies displacing each pixel in it.
In our image, we define 6 source and destination points labelled "1-6": "1-4" in the image
corners, "5" near the smile corner and "6" in an eye. At the corners, there are no
displacements. The smile corner moved upward and the eye position moved down.

Thin Plate Splines provides a very handy interpolator for image deformation.
"""
import matplotlib.pyplot as plt
import numpy as np

import skimage as ski

astronaut = ski.data.astronaut()

src = np.array([[50, 50], [400, 50], [50, 400], [400, 400], [240, 150], [200, 100]])
dst = np.array([[50, 50], [400, 50], [50, 400], [400, 400], [276, 100], [230, 100]])

# Fit the thin plate spline from output to input

warped_img = ski.transform.tps_warp(
    astronaut, src[:, ::-1], dst[:, ::-1], grid_scaling=1
)


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
# each target point. These coefficients are then used to transform an arbitrary
# point associated with the reference to an interpolated location on the target.

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
trans = ski.transform.TpsTransform()
trans.estimate(source_xy, target_xy)


# Create another arbitrary point
samp2 = np.linspace(-1.8, 1.8, 10)
test_xy = np.tile(samp2, [2, 1]).T

# Estimate transformed points from given sets of source and targets points
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
