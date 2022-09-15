"""
========================================
Track solidification of a metallic alloy
========================================

In this example, we identify and track the solid-liquid (S-L) interface in a
nickel-based alloy undergoing solidification. Tracking the solidification over
time enables the calculatation of the solidification velocity. This is
important to characterize the solidified structure of the sample and will be
used to inform research into additive manufacturing of metals. The image
sequence was obtained by the Center for Advanced Non-Ferrous Structural Alloys
(CANFSA) using synchrotron x-radiography at the Advanced Photon Source (APS)
at Argonne National Laboratory (ANL). This analysis was first presented at
a conference [1]_.

.. [1] Corvellec M. and Becker C. G. (2021, May 17-18)
       "Quantifying solidification of metallic alloys with scikit-image"
       [Conference presentation]. BIDS ImageXD 2021 (Image Analysis Across
       Domains). Virtual participation.
       https://www.youtube.com/watch?v=cB1HTgmWTd8
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial
import plotly.io
import plotly.express as px
import pandas as pd

from skimage import color, draw, filters, measure, restoration
from skimage.data import nickel_solidification

image_sequence = nickel_solidification()

y0, y1, x0, x1 = 0, 180, 100, 330

image_sequence = image_sequence[:, y0:y1, x0:x1]

print(f'shape: {image_sequence.shape}')

#####################################################################
# The dataset is a 2D image stack with 11 frames (time points).
# In the first section of this tutorial, image processing steps
# reducing noise in the images are performed on the entire 3D dataset
# such that the removal over localized, transient noise is favored as
# opposed to physical features (e.g. bubbles, splatters, etc.) that
# exist in roughly the same position from one frame to the next.

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
# Let us first apply a Gaussian low-pass filter in order to smooth
# the images and reduce noise.
# Next, we compute the image deltas, i.e., the sequence of differences
# between two consecutive frames. To do this, we subtract the image sequence
# from itself, but offset by one frame so that the subtracted images are
# one frame behind in time.

smoothed = filters.gaussian(image_sequence)
image_deltas = smoothed[1:, :, :] - smoothed[:-1, :, :]

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
# S-L interface). We now apply a total variation denoising filter to reduce
# noise beyond the interface.

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
# Our next step is to create binary images, splitting the images
# into foreground and background: We want the S-L interface
# to be the most prominent feature in the foreground of each binary image,
# so that it can eventually be separated from the rest of the image.
#
# We need
# a threshold value ``thresh_val`` to create our binary images, ``binarized``.
# This can be set manually, but we shall use an automated minimum threshold
# method from the ``filters`` submodule of scikit-image (there are other
# methods that may work better for different applications).

thresh_val = filters.threshold_minimum(denoised)
binarized = denoised > thresh_val

fig = px.imshow(
    binarized,
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': 'time point'}
)
plotly.io.show(fig)

#####################################################################
# Select largest region
# =====================
# In our binary images, the S-L interface appears as the largest region of
# connected pixels. For this step of the workflow, we will operate on each
# 2D image separately, as opposed to the entire 3D dataset. We are
# only interested in regions connected in space, so we operate
# on one image at a time so the regions do not span multiple moments in
# time across multiple images. We do this with a list comprehension by labeling each binarized
# image with the function :func:`skimage.measure.label()` while slicing the dataset from
# ``i = 0`` to ``i = binarized.shape[0]``.
# We can do this first labeling each separate
# region of connected pixels in the binary images
# The labels can be visualized by overlaying
# ``labeled`` over the original images (``image_sequence``). The function
# :func:`skimage.color.label2rgb()` takes a 2D image, so we must be
# careful to account for offset introduced after the image delta step when
# plotting these images.

labeled_list = [
        measure.label(binarized[i, :, :]) for i in range(binarized.shape[0])]
labeled_overlay_0 = color.label2rgb(
        labeled_list[0], image=image_sequence[1, :, :], bg_label=0)

fig = px.imshow(labeled_overlay_0, color_continuous_scale='gray')
plotly.io.show(fig)

#####################################################################
# We will now select the largest region in each image. We can do this
# by computing region properties, including the ``area`` property, and
# sorting by ``area`` values. Function
# :func:`skimage.measure.regionprops_table()` returns a table of region
# properties which can be readily read into a Pandas ``DataFrame``.

props_0 = measure.regionprops_table(
        labeled_list[0], properties=('label', 'area', 'bbox'))
props_0_df = pd.DataFrame(props_0)
props_0_df = props_0_df.sort_values('area', ascending=False)
# Show top five rows
props_0_df.head()

#####################################################################
# We can select the region from the ``labeled`` image by selecting all
# pixels that match the region's label. We will do this for each image
# by iterating through ``labeled_list``. The label of the largest region
# in each labeled image is retrieved (after creating and sorting a
# new Dataframe ``props_df`` for each image) by selecting the 0th item from the
# ``'label'`` column of that Dataframe. We'll also store the bounding box,
# or bbox, information for each image, which will be used to track the
# position of the S-L interface.

largest_mask_list = []
bboxes = []
for labeled in labeled_list:
    props = measure.regionprops_table(
            labeled, properties=('label', 'area', 'bbox'))
    props_df = pd.DataFrame(props)
    # Sort properties table based on the area column in descending order
    # (largest area will be in row 0)
    props_df = props_df.sort_values('area', ascending=False)
    # Append binary image with only region with the largest area
    largest_mask_list.append(labeled == props_df.iloc[0]['label'])
    bboxes.append([props_df.iloc[0][f'bbox-{i}'] for i in range(4)])

# Stack list of 2D arrays into 3D array
largest_masked = np.stack(largest_mask_list)
fig = px.imshow(
    largest_masked,
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': 'time point'}
)
plotly.io.show(fig)

#####################################################################
# To visualize the bbox, we create a 4D image to introduce
# RGB color channels. After initializing an empty array, we broadcast
# the region mask to each RGB channel so the interface region appears white.
# We then use the function :func:`skimage.draw.rectangle_perimeter` to
# generate the coordinates of a rectangle to overlay on the image.

largest_masked_color = np.zeros((*largest_masked.shape, 3))
# Iterate through bboxes and largest_mask_list at the same time
for i, (bbox, mask) in enumerate(zip(bboxes, largest_mask_list)):
    # Broadcast the mask to each RGB channel so region appears white
    for channel in range(3):
        largest_masked_color[i, :, :, channel] = mask
    minr, minc, maxr, maxc = bbox
    rect_pts_r, rect_pts_c = draw.rectangle_perimeter(
            (minr, minc), (maxr, maxc))
    # Add rectangle coords to channel 0 so rectangle appears red
    largest_masked_color[i, rect_pts_r, rect_pts_c, 0] = 1
fig = px.imshow(
    largest_masked_color,
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': 'time point'}
)
plotly.io.show(fig)
