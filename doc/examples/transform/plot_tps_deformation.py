r"""
==========================================
Interpolate images with thin-plate splines
==========================================

Thin-plate splines (TPS) refer to a method for interpolating data [1]_.
In an image context, we use a 2D generalization of TPS, i.e., a two-dimensional
simulation of a 1D cubic spline, which is the solution of the biharmonic equation [2]_.
According to [3]_, the basic solution of the biharmonic function was expanded.
Given corresponding source and target control points, TPS is used to transform a space.
In this case our space is a 2D image.

It should be noted that the commonly used in image processing
Affine Transformation [4]_ can be understood as special variant of TPS.

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

.. [4] Wikipedia, Affine transformation
       https://en.wikipedia.org/wiki/Affine_transformation#Image_transformation


Image Interpolation
===================

In this example we will see how to use thin plate spline interpolation in the
context of interpolating the correspondence of five pairs of landmarks.
The examples shows landmarks before and after being bent using the TPS algorithm,
and the control points (X) used as input the the algorithm.
"""
import matplotlib.pyplot as plt
import numpy as np

import skimage as ski

chess = ski.data.checkerboard()

src = np.array([[3.6929, 10.3819],[6.5827, 8.8386], [6.7766, 12.0866], [4.8189, 11.2047], [5.6969, 10.0748]])
dst = np.array([[3.9724, 6.5354], [6.6969, 4.1181], [6.5394, 7.2362], [5.4016, 6.4528], [5.7756, 5.1142]])

src *= chess.shape[0]//15
dst *= chess.shape[0]//15
# Fit the thin plate spline from output to input
tps = ski.transform.TpsTransform()
warped_img = ski.transform.tps_warp(chess, src, dst, grid_scaling=1)


fig, axs = plt.subplots(1, 2, figsize=(16, 8))
# axs[0].axis('off')
# axs[1].axis('off')

labels = ['1', '2', '3', '4', '5']  # Adjust the number of labels to match the number of points

axs[0].imshow(chess[..., ::-1], origin='upper', cmap='gray')
axs[0].scatter(src[:, 0], src[:, 1] , marker='x', color='green')

for i, label in enumerate(labels):
    axs[0].annotate(label, (src[:, 0][i], src[:, 1][i] ),
                    textcoords="offset points", xytext=(0, 10), ha='center', color='red')

axs[1].imshow(warped_img[..., ::-1], origin='upper', cmap='gray')
axs[1].scatter(dst[:, 0] , dst[:, 1] , marker='x', color='green')

for i, label in enumerate(labels):
    axs[1].annotate(label, (dst[:, 0][i] , dst[:, 1][i] ),
                    textcoords="offset points", xytext=(0, 10), ha='center', color='red')

plt.show()

######################################################################
#
# Deformation
# ===========
# In this example thin-plate spline is applied to source coordinates and to
# each target coordinates to derive an interpolation function and coefficients for
# each target points. These coefficients is then used to transforms an arbitrary
# point associated with the reference to an interpolated location on the target.

import matplotlib.pyplot as plt

import skimage as ski

samp = np.linspace(-2, 2, 4)
xx, yy = np.meshgrid(samp, samp)

# Make source points
source_xy = np.column_stack((xx.ravel(), yy.ravel()))

# Make target points
yy[:, [0, 3]] *=2
target_xy = np.column_stack((xx.ravel(), yy.ravel()))


# Compute the coefficient
trans = ski.transform.TpsTransform()
trans.estimate(source_xy, target_xy)


# Make another arbitiary point
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
ax1.scatter(transformed_xy[:,0], transformed_xy[:,1], c="orange", label='Transformed Test Points')
ax1.legend(loc="upper center")
ax1.set_title("Target and Transformed Coordinates")
plt.show()
