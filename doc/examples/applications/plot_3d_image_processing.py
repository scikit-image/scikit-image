"""
============================
Analyze 3D images (of cells)
============================

This tutorial is an introduction to three-dimensional image processing. Images
are represented as `numpy` arrays. A single-channel, or grayscale, image is a
2D matrix of pixel intensities of shape `(row, column)`. We can construct a 3D
volume as a series of 2D planes, giving 3D images the shape
`(plane, row, column)`. Multichannel images have an additional channel
dimension in the final position containing color information.

Some 3D images are constructed with equal resolution in each dimension (e.g.,
computer-generated rendering of a sphere). Most experimental data are captured
with a lower resolution in one of the three dimensions, e.g., photographing
thin slices to approximate a 3D structure as a stack of 2D images.
The distance between pixels in each dimension, called spacing, is encoded as a
tuple and is accepted as a parameter by some `skimage` functions and can be
used to adjust contributions to filters.

The data used in this tutorial were provided by the Allen Institute for Cell
Science. They were downsampled by a factor of 4 in the `row` and `column`
dimensions to reduce computational time.

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

#####################################################################
# The distance between pixels was reported by the microscope used to image the
# cells.
