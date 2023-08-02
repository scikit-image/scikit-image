r"""
==========================================
Interpolate images with thin-plate splines
==========================================

Thin-plate splines (TPS) refer to a method for interpolating data [1]_.
In an image context, we use a 2D generalization of TPS, i.e., a two-dimensional
simulation of a 1D cubic spline, which is the solution of the biharmonic equation [2]_.
According to [3]_, the basic solution of the biharmonic function was expanded.
Given corresponding source and target points, TPS is used to compute
a spatial deformation function for every point in the 2D plane or 3D volume.

It should be noted that the commonly used in image processing
Affine Transformation [4]_ can be understood as special variant of TPS.

This script shows how to use the Thine Plate Spine Transformation for
image warping.

For further information on TPS Transformation, see:

.. [1] Wikipedia, Cubic Spline interpolation
       https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation

.. [2] Weisstein, Eric W. "Biharmonic Equation." From MathWorld--A Wolfram Web Resource
       https://mathworld.wolfram.com/BiharmonicEquation.html

.. [3] Bookstein, Fred L. "Principal warps: Thin-plate splines and the
       decomposition of deformations." IEEE Transactions on pattern analysis and
       machine intelligence 11.6 (1989): 567–585.

.. [4] Wikipedia, Affine transformation
       https://en.wikipedia.org/wiki/Affine_transformation#Image_transformation


Image Interpolation
===================

In this example we will see how to use thin plate spline interpolation in the
context of image interpolation based on landmarks or control points.

"""
import matplotlib.pyplot as plt
import numpy as np

import skimage as ski

src = np.array([[0.25, 0.25],
                [0.25, 0.75],
                [0.75, 0.25],
                [0.75, 0.75]])

dst = np.array([[0.35, 0.35],
                [0.35, 0.65],
                [0.65, 0.35],
                [0.65, 0.65]])

astronaut = ski.data.astronaut()
width, height, _ = astronaut.shape
start = (50, 150)
end = (250, 350)
rr, cc = ski.draw.rectangle_perimeter(start=start, end=end, shape=astronaut.shape, clip=True)
astronaut[rr, cc] = 1

src *= [height, width]
dst *= [height, width]

# Fit the thin plate spline from output to input
tps = ski.transform.TpsTransform()

warped = ski.transform.tps_warp(astronaut, src, dst)

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(astronaut)
ax[1].imshow(warped)
plt.show()



######################################################################
#
# Deformation
# ===========
# In 2D cases, given a set of K corresponding points, the TPS warp is described by
# `2(K+3)`` parameters which include 6 global affine motion parameters and
# `2K` coefficients for correspondences of the control points.
#
# In this example we compute the coefficients, and then transform an arbitrary
# point from source surface to the deformed surface.

import matplotlib.pyplot as plt

import skimage as ski

samp = np.linspace(-2, 2, 4)
xx, yy = np.meshgrid(samp, samp)

# Make source points
source_xy = np.column_stack((xx.ravel(), yy.ravel()))

# Make target points
yy[:, [0, 3]] *=2
target_xy = np.column_stack((xx.ravel(), yy.ravel()))

# Get the coeefficient
trans = ski.transform.TpsTransform()
trans.estimate(source_xy, target_xy)

# Make an arbitiary point
samp2 = np.linspace(-1.8, 1.8, 10)
test_xy = np.tile(samp2, [2, 1]).T

# Estimate transformed points from given sets of source and targets points.
transformed_xy = trans(test_xy)


fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

ax0.scatter(source_xy[:, 0], source_xy[:, 1], label='Source Points')
ax0.scatter(test_xy[:, 0], test_xy[:, 1], c="orange", label='Test Points')
ax0.legend(loc="upper center")
ax0.set_title('Source and Test Coordinates')

ax1.scatter(target_xy[:, 0], target_xy[:, 1], label='Target Points')
ax1.scatter(transformed_xy[0], transformed_xy[1], c="orange", label='Transformed Test Points')
ax1.legend(loc="upper center")
ax1.set_title("Target and Transformed Coordinates")
plt.show()
