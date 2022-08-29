"""
========================================
Track solidification of a metallic alloy
========================================

In this example, we identify and track the solid-liquid (S-L) interface in a
metallic sample undergoing solidification. Tracking the solidification over 
time enables the calculatation of the solidification velocity. This is 
important to characterize the solidified structure of the sample and will be 
used to inform research into additive manufacturing of metals. The image 
sequence was obtained by the Center for Advanced Non-Ferrous Structural Alloys 
(CANFSA) using synchrotron x-radiography at the Advanced Photon Source (APS) at Argonne 
National Laboratory (ANL). This analysis was presented at a conference [1]_.

.. [1] Corvellec M. and Becker C. G. (2021, May 17-18)
       "Quantifying solidification of metallic alloys with scikit-image"
       [Conference presentation]. BIDS ImageXD 2021 (Image Analysis Across
       Domains). Virtual participation.
       https://www.youtube.com/watch?v=cB1HTgmWTd8
"""

import numpy as np
import plotly.io
import plotly.express as px

from skimage import filters, measure, restoration, segmentation
from skimage.data import nickel_solidification

image_sequence = nickel_solidification()

y1 = 0
y2 = 180
x1 = 100
x2 = 330

image_sequence = image_sequence[:, y1:y2, x1:x2]

print(f'shape: {image_sequence.shape}')

#####################################################################
# The dataset is a 2D image stack with 11 frames (time points).

fig = px.imshow(
    image_sequence,
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': 'time point'}
)
plotly.io.show(fig)

#####################################################################
# Compute image deltas
# ====================
# Let us apply a Gaussian low-pass filter to the images in order to smooth
# the images and reduce noise.
# Next, we compute the image deltas, i.e., the sequence of differences
# between two consecutive frames. To do this, we subtract ``image_sequence``
# from itself, but offset by one frame so that the subtracted images are 
# one frame behind in time.

images_smoothed = filters.gaussian(image_sequence)
image_deltas = images_smoothed[1:, :, :] - images_smoothed[:-1, :, :]

fig = px.imshow(
    image_deltas,
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': 'time point'}
)
plotly.io.show(fig)

#####################################################################
# Clip lowest and highest intensities
# ===================================
# We now calculate the 5th and 95th percentile intensities of ``image_deltas``:
# We want to clip the intensities which lie below the 5th percentile
# intensity and above the 95th percentile intensity, while also rescaling 
# the intensity values to [0, 1]. 

p_low, p_high = np.percentile(image_deltas, [5, 95])
clipped = image_deltas - p_low
clipped[clipped < 0.0] = 0.0
clipped = clipped / p_high
clipped[clipped > 1.0] = 1.0

fig = px.imshow(
    clipped,
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': 'time point'}
)
plotly.io.show(fig)

#####################################################################
# Invert and denoise
# ==================
# Next, we invert the ``clipped`` images so the regions of highest intensity
# will correspond to the region we are interested in tracking (i.e., the 
# S-L interface). With the images inverted, we'll apply a total 
# variation denoising filter to reduce noise beyond the interface.

inverted = 1 - clipped
denoised = restoration.denoise_tv_chambolle(inverted, weight=0.15)

fig = px.imshow(
    denoised,
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': 'time point'}
)
plotly.io.show(fig) 

#####################################################################
# Binarize
# ========
# Our next step is to create binary images, splitting each image 
# into a foreground and a background: We want the solid-liquid interface 
# to be the most prominent feature in the foreground of each binary image,
# so that it can eventually be separated from the rest of the image.
# 
# We need 
# a threshold value ``thresh_val`` to create our binary images, ``mask``. 
# This can be set manually, but we'll use an automated minimum threshold 
# method from the ``filters`` submodule of scikit-image (there are other 
# methods that may work better for different applications).

thresh_val = filters.threshold_minimum(denoised)
mask = denoised > thresh_val

fig = px.imshow(
    mask,
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': 'time point'}
)
plotly.io.show(fig)

#####################################################################
# Filter minimum size
# ===================
