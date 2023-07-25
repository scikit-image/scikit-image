r"""
===============
TPS Deformation
===============

Inspired from https://github.com/tzing/tps-deformation
"""
import matplotlib.pyplot as plt
import numpy as np

import skimage as ski

samp = np.linspace(-2, 2, 4)
xx, yy = np.meshgrid(samp, samp)

# Make source points
source_xy = np.column_stack((xx.ravel(), yy.ravel()))

# Make target points
yy[:, [0, 3]] *=2
target_xy = np.column_stack((xx.ravel(), yy.ravel()))

# Get coefficient, use class
trans = ski.transform.TpsTransform()
trans.estimate(source_xy, target_xy)

# Make another arbitiary point
samp2 = np.linspace(-1.8, 1.8, 10)
test_xy = np.tile(samp2, [2, 1]).T

# Estimate transformed points from given sets of source and targets points.
transformed_xy = trans(test_xy[:,0], test_xy[:,1])


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
