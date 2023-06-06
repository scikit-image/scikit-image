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
       `Optical Coherence Tomography <https://eyewiki.aao.org/Optical_Coherence_Tomography#:~:text=3%20Limitations-,Overview,at%20least%2010%2D15%20microns.>`__,
       American Academy of Ophthalmology.
.. [2] Jules Scholler (2019) "Image denoising using inpainting"
       https://www.jscholler.com/2019-02-28-remove-dots/

"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.io
import plotly.express as px

from skimage import (
    filters, morphology, restoration
)
from skimage.data import palisades_of_vogt


#####################################################################
# The dataset we are using in this example is an image sequence (a movie!) of
# human in-vivo tissue. Specifically, it shows the *palisades of Vogt* of a
# given cornea sample.

#####################################################################
# Load image data
# ===============

image_seq = palisades_of_vogt()

print(f'number of dimensions: {image_seq.ndim}')
print(f'shape: {image_seq.shape}')
print(f'dtype: {image_seq.dtype}')

#####################################################################
# The dataset is a timeseries of 60 frames (which are 2D images). We can
# visualize it by taking advantage of the ``animation_frame`` parameter in
# Plotly's ``imshow`` function.

# fig = px.imshow(
#     image_seq,
#     animation_frame=0,
#     height=500,
#     width=500,
#     binary_string=True,
#     labels={'animation_frame': 'time point'},
#     title='Sample of in-vivo human cornea'
# )
# plotly.io.show(fig)

#####################################################################
# Average over time
# =================
# First, we want to detect those dark spots where the data are lost. In
# technical terms, we want to *segment* the dark spots. We want to do so for
# all frames in the sequence. Unlike the actual data (signal), the dark spots
# do not move from one frame to the next; they are still. Therefore, we begin
# by computing the time average of the image sequence. We shall use this
# time-averaged image to segment the dark spots, the latter then standing out
# with respect to the background (blurred signal).

image_avg = np.mean(image_seq, axis=0)

print(f'shape: {image_avg.shape}')

fig = px.imshow(
    image_avg,
    width=500,
    height=500,
    binary_string=True,
    title='Time-averaged image'
)
plotly.io.show(fig)

#####################################################################
# Use local thresholding
# ======================
# To segment the dark spots, we use thresholding. The images we are working
# with have uneven illumination, which causes variations in the (absolute)
# intensities of the foreground and the background, from one region to another
# (distant) one. It is therefore more fitting to compute different threshold
# values across the image, one for each region. This is called adaptive (or
# local) thresholding, as opposed to the usual thresholding procedure which
# employs a single (global) threshold for all pixels in the image.
#
# Let us compare the visibility of the hidden data in two different
# thresholding masks, one of which has the `block_size` set to 21, while
# the other has it set to 35. For this, we start by defining a convenience
# function to create a mask:

def create_mask(image_avg, spot_size):
    thresh_value = filters.threshold_local(
        image_avg,
        block_size=spot_size,
        offset=10
    )
    mask = (image_avg > thresh_value)
    return mask

#####################################################################
# Let's also define a function to display two plots side-by-side so
# that it is easier for us to draw comparisons between plots:

def plot_comparison(plot1, plot2, title1, title2):
    fig, (ax1, ax2) = plt.subplots(
        ncols=2,
        figsize=(12, 8),
        sharex=True,
        sharey=True
    )
    ax1.imshow(plot1)
    ax1.set_title(title1)
    ax1.axis('off')
    ax2.imshow(plot2)
    ax2.set_title(title2)
    ax2.axis('off')

#####################################################################
# Now, we plot the two masks. It seems that the dust-covered
# spots appear more distinct in the second mask!

mask1 = create_mask(image_avg, 21)
mask2 = create_mask(image_avg, 35)

plot_comparison(mask1, mask2, "spot_size = 21", "spot_size = 35")

#####################################################################
# Remove fine-grained features
# ============================
# We use diamond to create a diamond structuring element ``footprint`` for this example.
# Morphological ``opening`` on an image is defined as an *erosion followed by
# a dilation*. Opening can remove small bright spots (i.e. "dust") and
# connect small dark cracks.

footprint = morphology.diamond(1)
mask = morphology.opening(mask1, footprint)
plot_comparison(image_avg, mask, "original", "erosion")

#####################################################################
# Since *opening* an image starts with an erosion operation, bright regions
# which are smaller than the structuring element are removed.
# Let us try with a larger structuring element:

footprint = morphology.diamond(3)
mask = morphology.dilation(mask1, footprint)
plot_comparison(image_avg, mask,"original", "dilation")

#####################################################################
# Dilation enlarges bright regions and shrinks dark regions.
# Notice how the white spots of the image thickens, or gets dilated, as we increase the
# size of the diamond.

#####################################################################
# Apply mask across frames
# ========================
# Although masks are binary, they can be applied to images to filter out
# pixels where the mask is ``False``.
# Numpy's ``where()`` is a flexible way of applying masks.
# The application of a mask to the input image produces an output image of
# the same size as the input.
# Let's apply the mask across frame

image_masked = np.where(image_avg, mask, 0)
plot_comparison(image_avg, image_masked, "original", "Masked across frames")

#####################################################################
# Inpaint each frame separately
# =============================
# We are now ready to apply inpainting to each frame. For this we use function
# ``inpaint_biharmonic`` from the ``restoration`` module.
# This function takes two arrays as inputs:
# The image to restore and the mask corresponding to the regions we want to
# inpaint.

image_denoised = np.zeros(image_seq.shape)

for i in range(image_seq.shape[0]):
    image_denoised[i] = restoration.inpaint_biharmonic(image_seq[i], mask)

#####################################################################
# Let us view the sequence of restored images, where the dark spots have been
# inpainted, using an algorithm based on biharmonic equations.

# fig = px.imshow(
#     image_denoised,
#     animation_frame=0,
#     height=500,
#     width=500,
#     binary_string=True,
#     labels={'animation_frame': 'time point'},
#     title='Sample of in-vivo human cornea restored with inpainting'
# )
# plotly.io.show(fig)
