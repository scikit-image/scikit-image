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
In other words, we port this workflow from ImageJ Macro to Python with
scikit-image.

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
    labels={'animation_frame': 'time point', 'facet_col': 'channel'}
)
plotly.io.show(fig)

#####################################################################
# To begin with, let us consider the first channel of the first image.

image_t_0_channel_0 = image_sequence[0, 0, :, :]

#####################################################################
# Segment the nucleus rim
# =======================
# Let us apply a Gaussian low-pass filter to this image in order to smooth it.
# Next, we segment the nuclei, finding the threshold between the background
# and foreground with Otsu's method: We get a binary image.

smooth = filters.gaussian(image_t_0_channel_0, sigma=1.5)

thresh_value = filters.threshold_otsu(smooth)
thresh = smooth > thresh_value

fill = ndi.binary_fill_holes(thresh)

#####################################################################
# Following the original workflow, let us remove objects which touch the image
# border. Here, we can see part of another nucleus in the bottom right-hand
# corner.

clear = segmentation.clear_border(fill)
clear.dtype

#####################################################################
# We compute both the morphological dilation of this binary image and its
# morphological erosion.

dilate = morphology.binary_dilation(clear)

erode = morphology.binary_erosion(clear)

#####################################################################
# Finally, we subtract the eroded from the dilated to get the nucleus rim.

mask = dilate.astype(int) - erode.astype(int)
mask.dtype

#####################################################################
# Let us visualize these processing steps in a sequence of subplots.

_, ax = plt.subplots(2, 4, figsize=(12, 6), sharey=True)

ax[0, 0].imshow(image_t_0_channel_0, cmap=plt.cm.gray)
ax[0, 0].set_title('a) Raw')

ax[0, 1].imshow(smooth, cmap=plt.cm.gray)
ax[0, 1].set_title('b) Blur')

ax[0, 2].imshow(thresh, cmap=plt.cm.gray)
ax[0, 2].set_title('c) Threshold')

ax[0, 3].imshow(fill, cmap=plt.cm.gray)
ax[0, 3].set_title('c-1) Fill in')

ax[1, 0].imshow(clear, cmap=plt.cm.gray)
ax[1, 0].set_title('c-2) Keep one nucleus')

ax[1, 1].imshow(dilate, cmap=plt.cm.gray)
ax[1, 1].set_title('d) Dilate')

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
_ = ax[1].axis('off')

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


def get_mask(im, sigma=1.5):
    im = filters.gaussian(im, sigma=sigma)
    thresh_value = filters.threshold_otsu(im)
    im = im > thresh_value
    im = ndi.binary_fill_holes(im)
    # Clear objects touching image border
    im = segmentation.clear_border(im)
    dilate = morphology.binary_dilation(im)
    erode = morphology.binary_erosion(im)
    mask = dilate.astype(int) - erode.astype(int)
    return mask


#####################################################################
# Let us compute the mask sequence corresponding to our image sequence.

mask_sequence = np.zeros_like(image_sequence[:, 0, :, :])

for i in range(image_sequence.shape[0]):
    # each mask gets a different label, running from 1 to 15
    mask_sequence[i, :, :] = get_mask(image_sequence[i, 0, :, :]) * (i + 1)

props = measure.regionprops_table(
    mask_sequence,
    intensity_image=image_sequence[:, 1, :, :],
    properties=('label', 'area', 'intensity_mean')
)
np.testing.assert_array_equal(props['label'], np.arange(15) + 1)

fluorescence_change = [props['area'][i] * props['intensity_mean'][i]
                       for i in range(image_sequence.shape[0])]

fluorescence_change /= fluorescence_change[0]  # normalization

_, ax = plt.subplots()
ax.plot(fluorescence_change, 'rs')
ax.grid()
ax.set_xlabel('Time point')
ax.set_ylabel('Normalized total intensity')
ax.set_title('Change in fluorescence intensity at the nuclear envelope')

plt.show()

#####################################################################
# Reassuringly, we find the expected result: The total fluorescence
# intensity at the nuclear envelope increases 1.3-fold in the initial five
# time points, and then becomes roughly constant.
