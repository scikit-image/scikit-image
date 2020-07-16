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
