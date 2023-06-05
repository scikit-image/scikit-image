"""
==============================================================
Reconstruct dust-obstructed human cornea image with inpainting
==============================================================

Optical Coherence Tomography (OCT) [1]_ is used to provide eye doctors with an
image of the retina in the back of a patient's eye.

Dust may stick to the reference mirror of the equipment, causing dark spots to
appear on the images. The problem is that these dark spots cover areas of
in-vivo tissue, hence hiding data of interest. Our goal here is to restore
(reconstruct) the hidden areas based on the pixels near their boundaries.

This tutorial is adapted from an application shared by Jules Scholler in [2]_.
The images were acquired by Viacheslav Mazlin.

.. [1] Vinay A. Shah, MD (2015)
       `Optical Coherence Tomography <https://eyewiki.aao.org/Optical_Coherence_Tomography#:~:text=3%20Limitations-,Overview,at%20least%2010%2D15%20microns.>`_,
       American Academy of Ophthalmology.
.. [2] Jules Scholler (2019) "Image denoising using inpainting"
       https://www.jscholler.com/2019-02-28-remove-dots/

"""

import imageio.v3 as iio
import numpy as np
import plotly.io
import plotly.express as px

from skimage import filters


#####################################################################
# The dataset we are using in this example is an image sequence showing the
# palisades of Vogt in a human cornea in vivo. Basically, it is a
# black-and-white movie!

#####################################################################
# Load image data
# ===============

image_seq = iio.imread('https://gitlab.com/mkcor/data/-/raw/70eb189f9b1c512fc8926891a2bdf96b67dcf441/in-vivo-cornea-spots.tif')

print(f'number of dimensions: {image_seq.ndim}')
print(f'shape: {image_seq.shape}')
print(f'dtype: {image_seq.dtype}')

#####################################################################
# The dataset is a timeseries of 60 (2D) images. We can visualize it by taking
# advantage of the `animation_frame` parameter in Plotly's `imshow` function.

fig = px.imshow(
    image_seq,
    animation_frame=0,
    height=500,
    width=500,
    binary_string=True,
    labels=dict(animation_frame='time point'),
    title='In-vivo human cornea'
)
plotly.io.show(fig)

#####################################################################
# Average over time
# =================

image_seq_mean = np.mean(image_seq, axis=0)

print(f'shape: {image_seq_mean.shape}')

fig = px.imshow(
    image_seq_mean,
    width=500,
    height=500,
    binary_string=True
)
plotly.io.show(fig)

#####################################################################
# Use local thresholding
# ======================

spot_size = 17

thresh_value = filters.threshold_local(
    image_seq_mean,
    block_size=spot_size
)

#####################################################################
# Remove fine-grained features
# ============================

#####################################################################
# Apply mask across frames
# ========================

#####################################################################
# Inpaint each frame separately
# =============================
