"""
============================
Explore 3D images (of cells)
============================

This tutorial is an introduction to three-dimensional image processing. Images
are represented as `numpy` arrays. A single-channel, or grayscale, image is a
2D matrix of pixel intensities of shape `(row, column)`. We can construct a 3D
volume as a series of 2D planes, giving 3D images the shape
`(plane, row, column)`. A multichannel, or RGB(A), image has an additional
`channel` dimension in the final position containing color information.

These conventions are summarized in the table below:

=============== ===============================
Image type      Coordinates
=============== ===============================
2D grayscale    `(row, column)`
2D multichannel `(row, column, channel)`
3D grayscale    `(plane, row, column)`
3D multichannel `(plane, row, column, channel)`
=============== ===============================

Some 3D images are constructed with equal resolution in each dimension (e.g.,
computer-generated rendering of a sphere). Most experimental data are captured
with a lower resolution in one of the three dimensions, e.g., photographing
thin slices to approximate a 3D structure as a stack of 2D images.
The distance between pixels in each dimension, called spacing, is encoded as a
tuple and is accepted as a parameter by some `skimage` functions and can be
used to adjust contributions to filters.

The data used in this tutorial were provided by the Allen Institute for Cell
Science. They were downsampled by a factor of 4 in the `row` and `column`
dimensions to reduce their size and, hence, computational time. The spacing
information was reported by the microscope used to image the cells.

"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy import ndimage as ndi

from skimage import (
    feature, filters, io, measure, morphology, segmentation, util
)
from skimage.data import image_fetcher


#####################################################################
# Load and display 3D images
# ==========================
# Three-dimensional data can be loaded with `io.imread`.

path = image_fetcher.fetch('data/cells.tif')
data = io.imread(path)

print("shape: {}".format(data.shape))
print("dtype: {}".format(data.dtype))
print("range: ({}, {})".format(data.min(), data.max()))

# Report spacing from microscope
original_spacing = np.array([0.2900000, 0.0650000, 0.0650000])

# Account for downsampling of slices by 4
rescaled_spacing = original_spacing * [1, 4, 4]

# Normalize spacing so that pixels are a distance of 1 apart
spacing = rescaled_spacing / rescaled_spacing[2]

print("microscope spacing: {}\n".format(original_spacing))
print("rescaled spacing: {} (after downsampling)\n".format(rescaled_spacing))
print("normalized spacing: {}\n".format(spacing))

#####################################################################
# Let us try and visualize the (3D) image with `io.imshow`.

try:
    io.imshow(data, cmap="gray")
except TypeError as e:
    print(str(e))

#####################################################################
# Function `io.imshow` can only display grayscale and RGB(A) 2D images.
# We can thus use it to visualize 2D planes. By fixing one axis, we can
# observe three different views of the image.

def show_plane(ax, plane, cmap="gray", title=None):
    ax.imshow(plane, cmap=cmap)
    ax.axis("off")

    if title:
        ax.set_title(title)

_, (a, b, c) = plt.subplots(ncols=3, figsize=(15, 5))

show_plane(a, data[32], title="Plane = 32")
show_plane(b, data[:, 128, :], title="Row = 128")
show_plane(c, data[:, :, 128], title="Column = 128")

#####################################################################
# As hinted before, a three-dimensional image can be viewed as a series of
# two-dimensional planes. Let us write a helper function, `display`, to
# display 30 planes of our data. By default, every other plane is displayed.

def display(im3d, cmap="gray", step=2):
    _, axes = plt.subplots(nrows=5, ncols=6, figsize=(16, 14))

    vmin = im3d.min()
    vmax = im3d.max()

    for ax, image in zip(axes.flatten(), im3d[::step]):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

display(data)

#####################################################################
# Alternatively, we can explore these planes (slices) interactively using
# Jupyter widgets. Let the user select which slice to display and show the
# position of this slice in the 3D dataset.

def slice_in_3D(ax, i):
    # From https://stackoverflow.com/questions/44881885/python-draw-3d-cube
    Z = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 1],
                  [1, 1, 1],
                  [0, 1, 1]])

    Z = Z * data.shape

    r = [-1, 1]

    X, Y = np.meshgrid(r, r)
    # Plot vertices
    ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

    # List sides' polygons of figure
    verts = [[Z[0], Z[1], Z[2], Z[3]],
             [Z[4], Z[5], Z[6], Z[7]],
             [Z[0], Z[1], Z[5], Z[4]],
             [Z[2], Z[3], Z[7], Z[6]],
             [Z[1], Z[2], Z[6], Z[5]],
             [Z[4], Z[7], Z[3], Z[0]],
             [Z[2], Z[3], Z[7], Z[6]]]

    # Plot sides
    ax.add_collection3d(
        Poly3DCollection(
            verts,
            facecolors=(0, 1, 1, 0.25),
            linewidths=1,
            edgecolors="darkblue"
        )
    )

    verts = np.array([[[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 1],
                       [0, 1, 0]]])
    verts = verts * (60, 256, 256)
    verts += [i, 0, 0]

    ax.add_collection3d(
        Poly3DCollection(
            verts,
            facecolors="magenta",
            linewidths=1,
            edgecolors="black"
        )
    )

    ax.set_xlabel("plane")
    ax.set_ylabel("row")
    ax.set_zlabel("col")

    # Autoscale plot axes
    scaling = np.array([getattr(ax,
        "get_{}lim".format(dim))() for dim in "xyz"])
    ax.auto_scale_xyz(* [[np.min(scaling), np.max(scaling)]] * 3)

def explore_slices(data, cmap="gray"):
    from ipywidgets import interact
    N = len(data)

    @interact(plane=(0, N - 1))
    def display_slice(plane=34):
        fig, ax = plt.subplots(figsize=(20, 5))

        ax_3D = fig.add_subplot(133, projection="3d")

        show_plane(ax, data[plane], title="Plane {}".format(plane), cmap=cmap)
        slice_in_3D(ax_3D, plane)

        plt.show()

    return display_slice

explore_slices(data);
