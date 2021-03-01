"""
==========================================
Interact with 3D images (of kidney tissue)
==========================================

In this tutorial, we explore interactively a biomedical image which has three
spatial dimensions and three colour dimensions (channels).
For a general introduction to 3D image processing, please refer to
:ref:`sphx_glr_auto_examples_applications_plot_3d_image_processing.py`.
The data we use here correspond to kidney tissue which Genevieve Buckley
imaged with confocal fluorescence microscopy (more details at [1]_ under
``kidney-tissue-fluorescence.tif``).

.. [1] https://gitlab.com/scikit-image/data/#data

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import plotly.express as px
from skimage import data


#####################################################################
# Load image
# ==========
# This biomedical image is available through `scikit-image`'s data registry.

data = data.kidney()

#####################################################################
# The returned dataset is a 3D multichannel image:

print(f'number of dimensions: {data.ndim}')
print(f'shape: {data.shape}')
print(f'dtype: {data.dtype}')

#####################################################################
# Dimensions are provided in the following order: ``(z, y, x, c)``.

n_plane, n_Y, n_X, n_chan = data.shape

#####################################################################
# Let us consider only a slice (2D plane) of the data for now. More
# specifically, let us consider the slice located halfway in the stack.
# The `imshow` function can display both grayscale and RGB(A) 2D images.

_, ax = plt.subplots()
ax.imshow(data[n_plane // 2])

#####################################################################
# According to the warning message, the range of values is unexpected. The
# image rendering is clearly not satisfactory colour-wise.

vmin, vmax = data.min(), data.max()
print(f'range: ({vmin}, {vmax})')

#####################################################################
# We turn to `plotly`'s implementation of the `imshow` function, for it
# supports `value ranges
# <https://plotly.com/python/imshow/#defining-the-data-range-covered-by-the-color-range-with-zmin-and-zmax>`_
# beyond ``(0.0, 1.0)`` for floats and ``(0, 255)`` for integers.

px.imshow(data[n_plane // 2], zmax=vmax)
# sphinx_gallery_thumbnail_number = 2

#####################################################################
# Here you go, *fluorescence* microscopy! Actually, kidney tissue is
# well-known for making a good microscope test slide, precisely because it has
# a great amount of auto-fluorescence: Everything will look fairly bright,
# regardless of the microscope settings.

#####################################################################
# Normalize range for each channel
# ================================
# Generally speaking, we may want to normalize the value range on a
# per-channel basis. Let us facet our data (slice) along the channel axis.
# This way, we get three single-channel images, where the max value of each
# image is used:

px.imshow(
    data[n_plane // 2],
    facet_col=2,
    binary_string=True,
    labels={'facet_col': 'channel'}
)

#####################################################################
# What is the range of values for each colour channel?

(vmin_0, vmin_1, vmin_2) = (data[:, :, :, 0].min(),
                            data[:, :, :, 1].min(),
                            data[:, :, :, 2].min())
(vmax_0, vmax_1, vmax_2) = (data[:, :, :, 0].max(),
                            data[:, :, :, 1].max(),
                            data[:, :, :, 2].max())
print(f'range for channel 0: ({vmin_0}, {vmax_0})')
print(f'range for channel 1: ({vmin_1}, {vmax_1})')
print(f'range for channel 2: ({vmin_2}, {vmax_2})')

#####################################################################
# Let us be very specific and pass value ranges on a per-channel basis:

px.imshow(
    data[n_plane // 2],
    zmin=[vmin_0, vmin_1, vmin_2],
    zmax=[vmax_0, vmax_1, vmax_2]
)

#####################################################################
# Plotly lets you interact with this visualization by panning, zooming in and
# out, let alone exporting the desired figure as a static image in PNG format.

#####################################################################
# Explore slices as animation frames
# ==================================
# Click the play button to move along the ``z`` axis, through the stack of all
# 16 slices.

px.imshow(
    data,
    zmin=[vmin_0, vmin_1, vmin_2],
    zmax=[vmax_0, vmax_1, vmax_2],
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': 'plane'}
)

#####################################################################
# Combine channel facetting and slice animation
# =============================================

px.imshow(
    data,
    animation_frame=0,
    facet_col=3,
    binary_string=True,
    labels={'facet_col': 'channel', 'animation_frame': 'plane'}
)

#####################################################################
# The biologist's eye can spot that the two bright blobs (best seen in
# ``channel=2``) are kidney glomeruli [2]_.
#
# .. [2] https://en.wikipedia.org/wiki/Glomerulus_(kidney)
