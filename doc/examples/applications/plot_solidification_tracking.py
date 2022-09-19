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
import pandas as pd
import plotly.express as px
import plotly.io

from skimage import color, draw, filters, measure, restoration
from skimage.data import nickel_solidification

image_sequence = nickel_solidification()

y0, y1, x0, x1 = 0, 180, 100, 330

image_sequence = image_sequence[:, y0:y1, x0:x1]

print(f'shape: {image_sequence.shape}')

#####################################################################
# The dataset is a 2D image stack with 11 frames (time points).
# We visualize and analyze it in a workflow where the first image processing
# steps are performed on the entire three-dimensional dataset (i.e., across
# space and time),
# such that the removal of localized, transient noise is favored as
# opposed to that of physical features (e.g., bubbles, splatters, etc.), which
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
# ending at the second-to-last frame from the image sequence starting
# at the second frame.

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
# 2D image separately, as opposed to the entire 3D dataset, because we are
# only interested in a single moment in time for each region.
#
# We select the largest region in each image
# by computing region properties, including the ``area`` property, and
# sorting by ``area`` values. Function
# :func:`skimage.measure.regionprops_table()` returns a table of region
# properties which can be read into a Pandas ``DataFrame``.

props_0 = measure.regionprops_table(
        measure.label(binarized[0, :, :]), properties=('label', 'area', 'bbox'))
props_0_df = pd.DataFrame(props_0)
props_0_df = props_0_df.sort_values('area', ascending=False)
# Show top five rows
props_0_df.head()

#####################################################################
# We can visualize the largest region in the 0th image with its
# bounding box (bbox) by first labeling the binary image and selecting
# the labels that correspond to the largest region.

labeled_0 = measure.label(binarized[0, :, :])
largest_region_0 = labeled_0 == props_0_df.iloc[0]['label']
minr, minc, maxr, maxc = [props_0_df.iloc[0][f'bbox-{i}'] for i in range(4)]
fig = px.imshow(largest_region_0, binary_string=True)
fig.add_shape(
        type="rect", x0=minc, y0=minr, x1=maxc, y1=maxr, line=dict(color="Red"))
plotly.io.show(fig)

#####################################################################
# Next we find the largest region in each image for all images in the
# dataset into a 3D NumPy array ``largest_region``. We'll also store
# the bbox information for each image, which will be used to track the
# position of the S-L interface.

largest_region = np.empty_like(binarized)
bboxes = []

for i in range(binarized.shape[0]):
    labeled = measure.label(binarized[i, :, :])
    props = measure.regionprops_table(
            labeled, properties=('label', 'area', 'bbox'))
    props_df = pd.DataFrame(props)
    props_df = props_df.sort_values('area', ascending=False)
    largest_region[i, :, :] = (labeled == props_df.iloc[0]['label'])
    bboxes.append([props_df.iloc[0][f'bbox-{i}'] for i in range(4)])
fig = px.imshow(
    largest_region,
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

largest_masked_color = np.zeros((*largest_region.shape, 3))
# Iterate through bboxes and largest_mask_list at the same time
for i, (bbox, mask) in enumerate(zip(bboxes, largest_region)):
    # Broadcast the mask to each RGB channel so region appears white
    largest_masked_color[i, :, :, :] = np.dstack([mask] * 3)
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

#####################################################################
# Plot interface location over time
# =================================
# The final step in this analysis is to plot the location of the S-L
# interfaces over time. This can be achieved by plotting ``maxr``
# over time since this value shows the y location of the bottom of
# the interface. The pixel size in this experiment was 1.93 microns per
# pixel and the framerate was 80,000 frames per second, so these values
# are used to convert pixels and image number to physical units.

ums_per_pixel = 1.93
fps = 80000
interface_y_um = [ums_per_pixel * bbox[2] for bbox in bboxes]
time_us = 1E6 / fps * np.arange(len(interface_y_um))
fig, ax = plt.subplots(dpi=100)
ax.scatter(time_us, interface_y_um)
ax.set_title('S-L interface location vs. time')
ax.set_ylabel('Location ($\mu$m)')
ax.set_xlabel('Time ($\mu$s)')
plt.show()

#####################################################################
# The solidification velocity can be obtained by fitting
# a linear polynomial to the scatter plot. The velocity is the first-order
# coefficient.

c0, c1 = polynomial.polyfit(time_us, interface_y_um, 1)
fig, ax = plt.subplots(dpi=100)
ax.scatter(time_us, interface_y_um)
ax.plot(time_us, c1 * time_us + c0, label=f'Velocity: {abs(round(c1, 3))} m/s')
ax.set_title('S-L interface location vs. time')
ax.set_ylabel('Location ($\mu$m)')
ax.set_xlabel('Time ($\mu$s)')
ax.legend()
plt.show()
