"""
==========================================
Estimate isotropy in a 3D microscopy image
==========================================

In this tutorial, we compute the structure tensor of a 3D image.
For a general introduction to 3D image processing, please refer to
:ref:`sphx_glr_auto_examples_applications_plot_3d_image_processing.py`.
The data we use here are sampled from an image of kidney tissue by Genevieve
Buckley in confocal fluorescence microscopy (more details at [1]_ under
``kidney-tissue-fluorescence.tif``).

.. [1] https://gitlab.com/scikit-image/data/#data

"""

import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px
from skimage import (
    data, feature
)


#####################################################################
# Load image
# ==========
# This biomedical image is available through `scikit-image`'s data registry.

data = data.kidney()

#####################################################################
# What exactly are the shape and size of our 3D multichannel image?

print(f'number of dimensions: {data.ndim}')
print(f'shape: {data.shape}')
print(f'dtype: {data.dtype}')

#####################################################################
# For the purposes of this tutorial, we shall consider only the second colour
# channel, which leaves us with a 3D single-channel image. What is the range
# of values?

n_plane, n_Y, n_X, n_chan = data.shape
v_min, v_max = data[:, :, :, 1].min(), data[:, :, :, 1].max()
print(f'range: ({v_min}, {v_max})')

#####################################################################
# Let us visualize the middle slice of our 3D image.

px.imshow(data[n_plane // 2, :, :, 1], zmin=v_min, zmax=v_max)

#####################################################################
# Let us pick a specific region, which shows relative X-Y isotropy. In
# contrast, the gradient is quite different (and, for that matter, weak) along
# Z.

sample = data[5:13, 380:410, 370:400, 1]


def display(im3d, cmap='gray', step=3):
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))

    it = 0
    for ax, image in zip(axes.flatten(), im3d[::step]):
        ax.imshow(image, cmap=cmap, vmin=v_min, vmax=v_max)
        ax.set_title(f'Plane = {5 + it * step}')
        ax.set_xticks([])
        ax.set_yticks([])
        it += 1


display(sample)

#####################################################################
# To view the sample data in 3D, run the following code:

# import plotly.graph_objects as go
#
# (n_Z, n_Y, n_X) = sample.shape
# Z, Y, X = np.mgrid[:n_Z, :n_Y, :n_X]
#
# fig = go.Figure(
#     data=go.Volume(
#         x=X.flatten(),
#         y=Y.flatten(),
#         z=Z.flatten(),
#         value=sample.flatten(),
#         opacity=0.5,
#         slices_z=dict(show=True, locations=[4])
#     )
# )
# fig.show()

#####################################################################
# Compute structure tensor
# ========================
# About the brightest region (i.e., at X ~ 22 and Y ~ 17), we can see
# variations (and, hence, strong gradients) along either X or Y over a typical
# length of 3. Note that, in the interactive plot below, `x` (resp. `y`)
# corresponds to Y (resp. X).

px.imshow(
    sample[0, :, :],
    zmin=v_min,
    zmax=v_max,
    title='Interactive view of bottom slice of sample data.'
)

#####################################################################
# Therefore, we choose a 'width' of 3 for the window function.

sigma = 3
A_elems = feature.structure_tensor(sample, sigma=sigma)

#####################################################################
# We can then compute the eigenvalues of the structure tensor.

eigen = feature.structure_tensor_eigenvalues(A_elems)
eigen.shape

#####################################################################
# Where is the largest eigenvalue?

coords = np.where(eigen == eigen.max())
assert coords[0] == 0  # by definition
coords

#####################################################################
# We are looking at a local property. Let us consider a tiny neighbourhood
# of this maximum in the X-Y plane.

eigen[0, coords[1], 20:23, 12:14]

#####################################################################
# If we examine the second-largest eigenvalues in this neighbourhood, we can
# see that they have the same order of magnitude as the largest ones.

eigen[1, coords[1], 20:23, 12:14]

#####################################################################
# In contrast, the third-largest eigenvalues are one order of magnitude
# smaller.

eigen[2, coords[1], 20:23, 12:14]

#####################################################################
# The region about ``coords[1:]`` is markedly anisotropic in 3D, as expected.
# We could say it is somewhat isotropic in the X-Y plane.
# In an ellipsoidal representation of the 3D structure tensor, we would get
# the pancake situation. The gradient directions are spread out (here, in the
# X-Y plane) and perpendicular to Z.
