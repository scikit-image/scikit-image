"""
=========================
Euler number
=========================

This example shows an illustration of the computation of the Euler number
in 2D and 3D objects.

For 2D objects, the Euler number is the number of objects minus the number of
holes. Notice that if an object is 8-connected, its complementary set is
4-connected, and conversely. It is also possible to compute the number of
objects using :func:`skimage.measure.label`, and to deduce the number of holes
from the difference between the two numbers.

For 3D objects, the Euler number is obtained as the number of objects plus the
number of holes, minus the number of tunnels, or loops. If an
object is 26-connected, its complementary set is 6-connected, and conversely.
The voxels are actually represented with blue transparent surfaces.
Inner porosities are represented in red.

"""
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import euler_number, label
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
SAMPLE = np.pad(SAMPLE, 1, mode='constant')

fig, ax = plt.subplots()
ax.imshow(SAMPLE, cmap=plt.cm.gray)
ax.axis('off')
e4 = euler_number(SAMPLE, connectivity=1)
object_nb_4 = label(SAMPLE, connectivity=1).max()
holes_nb_4 = object_nb_4 - e4
e8 = euler_number(SAMPLE, connectivity=2)
object_nb_8 = label(SAMPLE, connectivity=2).max()
holes_nb_8 = object_nb_8 - e8
ax.set_title('Euler number for N4: {} ({} objects, {} holes), \n for N8: {} ({} objects, {} holes)'.format(e4, object_nb_4, holes_nb_4, e8, object_nb_8, holes_nb_8))
plt.show()

######################################################################
# 3-D objects
# ===========
#
# In this example, a 3-D cube is generated, then holes and
# tunnels are added. Euler number is evaluated with 6 and 26 neighborhood
# configuration. This code is inpired by
# https://matplotlib.org/devdocs/gallery/mplot3d/voxels_numpy_logo.html


def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.grid(grid)
    ax.set_axis_off()
    return ax


def explode(data):
    """visualization to separate voxels
    """
    size = np.array(data.shape) * 2
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


def display_voxels(volume):
    """
    volume: (N,M,P) array
            Represents a binary set of pixels: objects are marked with 1,
            complementary (porosities) with 0.

    The voxels are actually represented with blue transparent surfaces.
    Inner porosities are represented in red.
    """

    # define colors
    red = '#ff0000ff'
    blue = '#1f77b410'

    # upscale the above voxel image, leaving gaps
    filled = explode(np.ones(volume.shape))

    fcolors = explode(np.where(volume, blue, red))

    # Shrink the gaps
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    # Define 3D figure and place voxels
    ax = make_ax()
    ax.voxels(x, y, z, filled, facecolors=fcolors)

    # Compute Euler number in 6 and 26 neighbourhood configuration, that
    # correspond to 1 and 3 connectivity, respectively
    e26 = euler_number(volume, connectivity=3)
    e6 = euler_number(volume, connectivity=1)
    plt.title('Euler number for N26: {}, for N6: {}'.format(e26, e6))
    plt.show()


# Define a volume of 7x7x7 voxels
n = 7
cube = np.ones((n, n, n), dtype=bool)
# Add a hole
c = int(n/2)
cube[c, c, c] = False
# Add a concavity that connects previous hole
cube[c, :, c] = False
# Add a new hole
cube[int(3*n/4), c-1, c-1] = False
# Add a hole in neighbourhood of previous one
cube[int(3*n/4), c, c] = False
# Add a tunnel
cube[:, c, int(3*n/4)] = False
display_voxels(cube)
