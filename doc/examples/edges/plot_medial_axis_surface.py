"""
==============
Thinning in 3D
==============

This example shows how to perfome medial axis and medial surface thinning
on 3D data.

The medial axis of an object is the set of all points having more than one
closest point on the object's boundary. It is often called the *topological
skeleton*, because it is a 1-pixel wide skeleton of the object, with the same
connectivity as the original object.

The medial surface is similar to the medial axis but instead returns a
1-pixel thick surface of the object, also preserving the connectivity.

Both thinning methods apply the methods described in [Lee94]_, which use an
octree data structure to examine a 3x3x3 neighborhood of a pixel. The
algorithm proceeds by iteratively sweeping over the image, and removing
pixels at each iteration until the image stops changing. Each iteration
consists of two steps: first, a list of candidates for removal is assembled;
then pixels from this list are rechecked sequentially, to better preserve
connectivity of the image.
"""


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from skimage import data
from skimage.morphology import medial_axis, medial_surface, skeletonize

# Generate 3D input data.
# The shape of the data closely matches the example shown in
# Fig. 12 in [Lee94]_.
data = np.zeros([30, 30, 30], dtype=np.uint8)
data[10:15, 5:10, 10:25] = 1
data[10:20, 10:20, 10:15] = 1
data[2:20, 20:25, 10:15] = 1
data[24:29, 5:20, 5:10] = 1
data[10:24, 15:20, 5:10] = 1

# Compute the medial axis.
mat = skeletonize(data, method='lee')

# Compute the medial surface.
mst = medial_surface(data)

# Visualize the results
fig = plt.figure(figsize=(12, 4))

ax = fig.add_subplot(1, 3, 1, projection=Axes3D.name)
ax.voxels(data)
ax.set_title('original')

ax = fig.add_subplot(1, 3, 2, projection=Axes3D.name)
ax.voxels(mat)
ax.set_title('medial axis')

ax = fig.add_subplot(1, 3, 3, projection=Axes3D.name)
ax.voxels(mst)
ax.set_title('medial surface')

fig.tight_layout()
plt.show()
