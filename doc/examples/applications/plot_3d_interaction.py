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
# We notice that the ranges of values is unusual: With images, we
# typically expect a range of ``(0.0, 1.0)`` for float values and a range of
# ``(0, 255)`` for integer values.
# Let us consider only a slice (2D plane) of the data for now. More
# specifically, let us consider the slice located halfway in the stack.
# The `imshow` function can display both grayscale and RGB(A) 2D images.

_, ax = plt.subplots()
ax.imshow(data[n_plane // 2])

#####################################################################
# The warning message echoes our concern, while the image rendering is clearly
# not satisfactory colour-wise. Not all multichannel images are RGB(A)!
# We turn to `plotly`'s implementation of the `imshow` function, for it lets
# us specify the `value range
# <https://plotly.com/python/imshow/#defining-the-data-range-covered-by-the-color-range-with-zmin-and-zmax>`_
# to map to a colour range.

px.imshow(
    data[n_plane // 2],
    zmin=[vmin_0, vmin_1, vmin_2],
    zmax=[vmax_0, vmax_1, vmax_2]
)
# sphinx_gallery_thumbnail_number = 2

#####################################################################
# Normalize range for each channel
# ================================

px.imshow(
    data[n_plane // 2],
    facet_col=2,
    binary_string=True,
    labels={'facet_col': 'channel'}
)

#####################################################################
# Explore slices as animation frames
# ==================================

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
