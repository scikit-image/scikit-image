"""
==========================================
Interact with 3D images (of kidney tissue)
==========================================

In this tutorial, we explore interactively a biomedical image which has three
spatial dimensions and four colour dimensions (channels).
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

from skimage import (
    data, exposure
)


#####################################################################
# Load image
# ==========

data = data.kidney()

print("shape: {}".format(data.shape))
print("dtype: {}".format(data.dtype))
print("range: ({}, {})".format(data.min(), data.max()))

#####################################################################
# We can visualize a slice (2D plane) of the data with the `io.imshow`
# function. By fixing one spatial axis, we can observe three different views.


def show_plane(ax, plane, title=None):
    ax.imshow(plane)
    ax.axis('off')

    if title:
        ax.set_title(title)


(n_plane, n_row, n_col, n_chan) = data.shape
_, (a, b, c) = plt.subplots(ncols=3, figsize=(15, 5))

show_plane(a, data[n_plane // 2], title=f'Plane = {n_plane // 2}')
show_plane(b, data[:, n_row // 2, :, :], title=f'Row = {n_row // 2}')
show_plane(c, data[:, :, n_col // 2, :], title=f'Column = {n_col // 2}')
