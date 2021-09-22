"""
======================================================
Measure fluorescence intensity at the nuclear envelope
======================================================

This example reproduces a well-established workflow in bioimage data analysis
for measuring the fluorescence intensity localized to the nuclear envelope, in
a time sequence of cell images (each with two channels and two spatial
dimensions) which shows a process of protein re-localization from the
cytoplasmic area to the nuclear envelope. This biological application was
first presented by Andrea Boni and collaborators in [1]_; it was used in a
textbook by Kota Miura [2]_ as well as in other works ([3]_, [4]_).
In other words, we port this workflow from ImageJ Macro to scikit-image.

.. [1] Boni A, Politi AZ, Strnad P, Xiang W, Hossain MJ, Ellenberg J (2015)
       "Live imaging and modeling of inner nuclear membrane targeting reveals
       its molecular requirements in mammalian cells" J Cell Biol
       209(5):705–720. ISSN: 0021-9525.
       :DOI:`10.1083/jcb.201409133`
.. [2] Miura K (2020) "Measurements of Intensity Dynamics at the Periphery of
       the Nucleus" in: Miura K, Sladoje N (eds) Bioimage Data Analysis
       Workflows. Learning Materials in Biosciences. Springer, Cham.
       :DOI:`10.1007/978-3-030-22386-1_2`
.. [3] Klemm A (2020) "ImageJ/Fiji Macro Language" NEUBIAS Academy Online
       Course: https://www.youtube.com/watch?v=o8tfkdcd3DA
.. [4] Vorkel D and Haase R (2020) "GPU-accelerating ImageJ Macro image
       processing workflows using CLIJ" https://arxiv.org/abs/2008.11799

"""

import imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly.io
import plotly.express as px
from scipy import ndimage as ndi

from skimage import (
    filters, measure, morphology, segmentation
)


#####################################################################
# We start with a single cell/nucleus to construct the workflow.

image_sequence = imageio.volread('http://cmci.embl.de/sampleimages/NPCsingleNucleus.tif')

print(f'shape: {image_sequence.shape}')

#####################################################################
# The dataset is a 2D image stack with 15 frames (time points) and 2 channels.

fig = px.imshow(
    image_sequence,
    facet_col=1,
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': 'time point', 'facet_col': 'channel'}
)
plotly.io.show(fig)

#####################################################################
# To begin with, let us consider the first channel of the first image.

image_t_0_channel_0 = image_sequence[0, 0, :, :]

#####################################################################
# Segment the nucleus rim
# =======================

smooth = filters.gaussian(image_t_0_channel_0, sigma=2)

thresh = smooth > 0.1

fill = ndi.binary_fill_holes(thresh)

#####################################################################
# Following the original workflow, let us remove objects which touch the image
# border. Here, we can see part of another nucleus in the bottom right-hand
# corner.

clear = segmentation.clear_border(fill)
label = measure.label(clear)

expand = segmentation.expand_labels(label, distance=4)

erode = morphology.erosion(label)

mask = expand - erode

_, ax = plt.subplots(2, 4, figsize=(12, 6), sharey=True)

ax[0, 0].imshow(image_t_0_channel_0, cmap=plt.cm.gray)
ax[0, 0].set_title('a) Raw')

ax[0, 1].imshow(smooth, cmap=plt.cm.gray)
ax[0, 1].set_title('b) Blur')

ax[0, 2].imshow(thresh, cmap=plt.cm.gray)
ax[0, 2].set_title('c) Threshold')

ax[0, 3].imshow(fill, cmap=plt.cm.gray)
ax[0, 3].set_title('c-1) Fill in')

ax[1, 0].imshow(label, cmap=plt.cm.gray)
ax[1, 0].set_title('c-2) Keep one nucleus')

ax[1, 1].imshow(expand, cmap=plt.cm.gray)
ax[1, 1].set_title('d) Expand')

ax[1, 2].imshow(erode, cmap=plt.cm.gray)
ax[1, 2].set_title('e) Erode')

ax[1, 3].imshow(mask, cmap=plt.cm.gray)
ax[1, 3].set_title('f) Nucleus Rim')

for a in ax.ravel():
    a.axis('off')

#####################################################################
# Apply the segmented rim as a mask
# =================================
# Now that we have segmented the nuclear membrane in the first channel, we use
# it as a mask to measure the intensity in the second channel.

image_t_0_channel_1 = image_sequence[0, 1, :, :]
selection = np.where(mask, image_t_0_channel_1, 0)

_, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

ax[0].imshow(image_t_0_channel_1)
ax[0].set_title('Second channel (raw)')
ax[0].axis('off')

ax[1].imshow(selection)
ax[1].set_title('Selection')
ax[1].axis('off')

#####################################################################
# Measure the total intensity
# ===========================
# The mean intensity is readily available as a region property in a labeled
# image.

props = measure.regionprops_table(
    mask,
    intensity_image=image_t_0_channel_1,
    properties=('label', 'area', 'intensity_mean')
)

#####################################################################
# We may want to check that the value for the total intensity

selection.sum()

#####################################################################
# can be recovered from:

props['area'] * props['intensity_mean']

#####################################################################
# Iterate the measurement for each time point
# ===========================================
# Let us write a function for the first processing step (segmentation of
# nucleus rim).


def get_mask(im, sigma=2, thresh=0.1, thickness=4):
    im = filters.gaussian(im, sigma=sigma)
    im = im > thresh
    im = ndi.binary_fill_holes(im)
    # Clear objects touching image border
    im = segmentation.clear_border(im)
    label = measure.label(im)
    expand = segmentation.expand_labels(label, distance=thickness)
    erode = morphology.erosion(label)
    mask = expand - erode
    return mask


#####################################################################
# Let us now loop through all the image frames.

fluorescence_change = []

for i in range(image_sequence.shape[0]):
    mask = get_mask(image_sequence[i, 0, :, :])
    props = measure.regionprops_table(
        mask,
        intensity_image=image_sequence[i, 1, :, :],
        properties=('label', 'area', 'intensity_mean')
    )
    assert props['label'] == 1
    intensity_total = props['area'] * props['intensity_mean']
    fluorescence_change.append(intensity_total)

fluorescence_change /= fluorescence_change[0]  # normalization

_, ax = plt.subplots()
ax.plot(fluorescence_change, 'rs')
ax.grid()
ax.set_xlabel('Time point')
ax.set_ylabel('Normalized total intensity')
ax.set_title('Change in fluorescence intensity at the nuclear envelope')

#####################################################################
# Reassuringly, we find the expected result: The total fluorescence
# intensity at the nuclear envelope increases 1.3-fold in the initial five
# time points, and then becomes roughly constant.

plt.show()
