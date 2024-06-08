"""
==========================================
Use thin-plate splines for image warping
==========================================

To warp an image, we start with a set of source and target coordinates.
The goal is to deform the image such that the source points move to the target
locations. Typically, we only know the target positions for a few, select
source points. To calculate the target positions for all other pixel positions,
we need a model. Various such models exist, such as `affine or projective
transformations <https://scikit-image.org/docs/stable/auto_examples/transform/plot_transform_types.html>`_.

Most transformations are linear (i.e., they preserve straight lines), but
sometimes we need more flexibility. One model that represents a non-linear
transformation, i.e. one where lines can bend, is thin-plate splines [1]_ [2]_.

Thin-plate splines draw on the analogy of a metal sheet, which has inherent
rigidity. Consider our source points: each has to move a certain distance, in
both the x and y directions, to land in their corresponding target positions.
First, examine only the x coordinates. Imagine placing a thin metal plate on
top of the image. Now bend it, such that at each source point, the plate's
z offset is the distance, positive or negative, that that source point has to
travel in the x direction in order to land in its target position. The plate
resists bending, and therefore remains smooth. We can read offsets for
coordinates other than source points from the position of the plate. The same
procedure can be repeated for the y coordinates.

This gives us our thin-plate spline model that maps any (x, y) coordinate to a
target position.

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

# Define a matching destination for each source point
src = np.array([[50, 50], [400, 50], [50, 400], [400, 400], [240, 150], [200, 100]])
dst = np.array([[50, 50], [400, 50], [50, 400], [400, 400], [276, 100], [230, 100]])

# Estimate the TPS transformation from these points and then warp the image.
# We switch `src` and `dst` here because `skimage.transform.warp` expects the
# inverse transformation!
tps = ski.future.ThinPlateSplineTransform()
tps.estimate(dst, src)
warped = ski.transform.warp(astronaut, tps)


fig, axs = plt.subplots(1, 2)

# Adjust the number of labels to match the number of points
labels = ["1", "2", "3", "4", "5", "9"]

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

axs[1].imshow(warped, cmap='gray')
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
# In this second example, we start with a set of source and target points.
# TPS are used to derive an interpolation function and coefficients from each
# of those points.
# These coefficients can then be used to translate another set of points, in
# the example below called "Original", to a new location matching the original
# deformation between source and target.

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
trans = ski.future.ThinPlateSplineTransform()
trans.estimate(source_xy, target_xy)

# Create another arbitrary point
samp2 = np.linspace(-1.8, 1.8, 10)
test_xy = np.tile(samp2, [2, 1]).T

# Estimate transformed points from given sets of source and target points
transformed_xy = trans(test_xy)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 3))

ax0.scatter(source_xy[:, 0], source_xy[:, 1], label='Source')
ax0.scatter(test_xy[:, 0], test_xy[:, 1], c='orange', label='Original')
ax0.legend(loc='upper center')
ax0.set_title('Source and original points')

ax1.scatter(target_xy[:, 0], target_xy[:, 1], label='Target')
ax1.scatter(
    transformed_xy[:, 0],
    transformed_xy[:, 1],
    c='orange',
    label='Transformed',
)
ax1.legend(loc="upper center")
ax1.set_title("Target and transformed points")
plt.show()
