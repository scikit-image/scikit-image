"""
=========================
Euler number
=========================

This example shows an illustration of the computation of the Euler number
in 2D and 3D objects.

For 2D objects, Euler number if the number of objects minus the number of 
holes. Notice that if an object is 8-connected, its complementary set is 
4-connected, and conversely.

For 3D objects, the number of tunnels has to be taken into account. If an 
object is 26-connected, its complementary set is 6-connected, and conversely.

"""
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import euler_number
import matplotlib.pyplot as plt
import numpy as np


# Sample image.
SAMPLE = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
     [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
     [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
)

plt.imshow(SAMPLE, cmap=plt.cm.gray)
e4 = euler_number(SAMPLE, 4)
e8 = euler_number(SAMPLE, 8)
plt.title('Euler number for N4: {}, for N8: {}'.format(e4, e8))
plt.show()

"""
3D objects. In order to visualize things, a cube is generated, then holes and
tunnels are added. Euler number is evaluated with 6 and 26 neighborhood 
configuration.
This code is inpired by 
https://matplotlib.org/devdocs/gallery/mplot3d/voxels_numpy_logo.html
"""

# defines a 3D figure


def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.grid(grid)
    ax.set_axis_off()
    return ax

# separate voxels


def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

# shrink the gaps between voxels


def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

# this function displays voxels in a transparent way, and porosities in red


def display_voxels(volume):
    # upscale the above voxel image, leaving gaps
    filled = explode(np.ones(volume.shape))
    fcolors = explode(np.where(volume, '#1f77b410', '#ff0000ff'))

    # Shrink the gaps
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    # Define 3D figure and place voxels
    ax = make_ax()
    ax.voxels(x, y, z, filled, facecolors=fcolors)

    # Compute Euler number in 6 and 26 neighbourhood configuration
    e26 = euler_number(volume, neighbourhood=26)
    e6 = euler_number(volume, neighbourhood=6)
    plt.title('Euler number for N26: {}, for N6: {}'.format(e26, e6))

    plt.show()


# Define a volume of 7x7x7 voxels
n = 7
cube = np.ones((n, n, n), dtype=bool)
display_voxels(cube)

# Add a hole
c = int(n/2)
cube[c, c, c] = False
display_voxels(cube)

# Add a concavity that connects previous hole
cube[c, 0:c, c] = False
display_voxels(cube)

# Add a new hole
cube[int(3*n/4), c-1, c-1] = False
display_voxels(cube)

# Add a hole in neighbourhood of previous one
cube[int(3*n/4), c, c] = False
display_voxels(cube)

# Add a tunnel
cube[:, c, int(3*n/4)] = False
display_voxels(cube)
