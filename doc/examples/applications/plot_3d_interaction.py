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
# The returned dataset is a 3D multichannel image with dimensions provided in
# ``(z, y, x, c)`` order.

print("shape: {}".format(data.shape))
print("dtype: {}".format(data.dtype))
v_min, v_max = data.min(), data.max()
print("range: ({}, {})".format(v_min, v_max))

#####################################################################
# First of all, we notice that the range of values is unusual: With images, we
# typically expect a range of ``(0.0, 1.0)`` for float values and a range of
# ``(0, 255)`` for integer values.
# Let us consider only a slice (2D plane) of the data for now. More
# specifically, let us consider the slice located halfway in the stack.
# The `imshow` function can display both grayscale and RGB(A) 2D images.

n_plane, n_row, n_col, n_chan = data.shape

_, ax = plt.subplots()
ax.imshow(data[n_plane // 2])

#####################################################################
# The warning message echoes our concern, while the image rendering is clearly
# not satisfactory colour-wise.
# We turn to `plotly`'s implementation of the `imshow` function, for it lets
# us specify the `value range
# <https://plotly.com/python/imshow/#defining-the-data-range-covered-by-the-color-range-with-zmin-and-zmax>`_
# to map to a colour range.

px.imshow(data[n_plane // 2], zmin=v_min, zmax=v_max)
# sphinx_gallery_thumbnail_number = 2
