"""
============================================
Restore spotted cornea image with inpainting
============================================

Optical coherence tomography (OCT) is a non-invasive imaging technique used by
ophthalmologists to take pictures of the back of a patient's eye [1]_.
When performing OCT,
dust may stick to the reference mirror of the equipment, causing dark spots to
appear on the images. The problem is that these dark spots cover areas of
in-vivo tissue, hence hiding data of interest. Our goal here is to restore
(reconstruct) the hidden areas based on the pixels near their boundaries.

This tutorial is adapted from an application shared by Jules Scholler [2]_.
The images were acquired by Viacheslav Mazlin.

.. [1] David Turbert, reviewed by Ninel Z Gregori, MD (2023)
       `What Is Optical Coherence Tomography?
       <https://www.aao.org/eye-health/treatments/what-is-optical-coherence-tomography>`_,
       American Academy of Ophthalmology.
.. [2] Jules Scholler (2019) "Image denoising using inpainting"
       https://www.jscholler.com/2019-02-28-remove-dots/
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.io
import plotly.express as px
import plotly.graph_objects as go
from scipy import sparse

from skimage import (
    filters, measure, morphology, restoration
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
# The dataset is an image stack with 60 frames (time points) and 2 spatial
# dimensions. Let us visualize 10 frames towards the beginning of the
# sequence, where we can see slight changes in illumination.
# We take advantage of the ``animation_frame`` parameter in
# Plotly's ``imshow`` function. As a side note, when the
# ``binary_string`` parameter is set to ``True``, the image is
# represented as grayscale.

fig = px.imshow(
    image_seq[12:22, :, :],
    animation_frame=0,
    height=500,
    width=500,
    binary_string=True,
    labels={'animation_frame': 'time point'},
    title='Sample of in-vivo human cornea'
)
plotly.io.show(fig)

#####################################################################
# Average over time
# =================
# First, we want to detect those dark spots where the data are lost. In
# technical terms, we want to *segment* the dark spots (for
# all frames in the sequence). Unlike the actual data (signal), the dark spots
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
# When calling the ``threshold_local`` function from the ``filters`` module,
# we may change the default neighborhood size (``block_size``), i.e., the
# typical size (number of pixels) over which illumination varies,
# as well as the ``offset`` (shifting the neighborhood's weighted mean).
# Let us try two different values for ``block_size``:

thresh_1 = filters.threshold_local(image_avg, block_size=21, offset=15)
thresh_2 = filters.threshold_local(image_avg, block_size=43, offset=15)

mask_1 = image_avg < thresh_1
mask_2 = image_avg < thresh_2

#####################################################################
# Let us define a convenience function to display two plots side by side, so
# it is easier for us to compare them:

def plot_comparison(plot1, plot2, title1, title2):
    fig, (ax1, ax2) = plt.subplots(
        ncols=2,
        figsize=(12, 8),
        sharex=True,
        sharey=True
    )
    ax1.imshow(plot1, cmap='gray')
    ax1.set_title(title1)
    ax1.axis('off')
    ax2.imshow(plot2, cmap='gray')
    ax2.set_title(title2)
    ax2.axis('off')

plot_comparison(mask_1, mask_2, "block_size = 21", "block_size = 43")

#####################################################################
# The "dark spots" appear to be more distinct in the second mask, i.e., the
# one resulting from using the larger ``block_size`` value.
# Regarding the value of the offset parameter, we noticed that increasing it
# instead of keeping its default zero value, would yield a more uniform
# background, hence letting the objects of interest stand out more visibly.
# Indeed:

thresh_0 = filters.threshold_local(image_avg, block_size=43)

mask_0 = image_avg < thresh_0

plot_comparison(mask_0, mask_2, "No offset", "offset = 15")

#####################################################################
# Remove fine-grained features
# ============================
# We use morphological filters to sharpen the mask and focus on the dark
# spots. The two fundamental morphological operators are *dilation* and
# *erosion*, where dilation (resp. erosion) sets the pixel to the brightest
# (resp. darkest) value of the neighborhood defined by a structuring element
# (footprint).
#
# Here, we use the ``diamond`` function from the ``morphology`` module to
# create a diamond-shaped footprint.
# An erosion followed by a dilation is called an *opening*.
# First, we apply an opening filter, in order to remove small objects and thin
# lines, while preserving the shape and size of larger objects.

footprint = morphology.diamond(1)
mask_open = morphology.opening(mask_2, footprint)
plot_comparison(mask_2, mask_open, "mask before", "after opening")

#####################################################################
# Since "opening" an image starts with an erosion operation, bright regions
# which are smaller than the structuring element have been removed.
# Let us try now with a larger footprint:

footprint = morphology.diamond(3)
mask_open = morphology.opening(mask_2, footprint)
plot_comparison(mask_2, mask_open, "mask before", "after opening")

#####################################################################
# Next, we can make the detected areas wider by applying a dilation filter:

mask_dilate = morphology.dilation(mask_open, footprint)
plot_comparison(mask_open, mask_dilate, "before", "after dilation")

#####################################################################
# Dilation enlarges bright regions and shrinks dark regions.
# Notice how, indeed, the white spots have thickened.

#####################################################################
# Inpaint each frame separately
# =============================
# We are now ready to apply inpainting to each frame. For this we use function
# ``inpaint_biharmonic`` from the ``restoration`` module. It implements an
# algorithm based on biharmonic equations.
# This function takes two arrays as inputs:
# The image to restore and a mask (with same shape) corresponding to the
# regions we want to inpaint.

image_seq_inpainted = np.zeros(image_seq.shape)

for i in range(image_seq.shape[0]):
    image_seq_inpainted[i] = restoration.inpaint_biharmonic(
        image_seq[i],
        mask_dilate
    )

#####################################################################
# Let us visualize one restored image, where the dark spots have been
# inpainted. First, we find the contours of the dark spots (i.e., of the mask)
# so we can draw them on top of the restored image;

contours = measure.find_contours(mask_dilate)

# Gather all (row, column) coordinates of the contours
x = []
y = []
for contour in contours:
    x.append(contour[:, 0])
    y.append(contour[:, 1])
# Flatten them
x_flat = np.concatenate(x).ravel()
y_flat = np.concatenate(y).ravel()
# Create a sparse matrix corresponding to the mask of these contours
data = np.ones(x_flat.shape, dtype='bool')
mtx = sparse.coo_matrix((data, (x_flat, y_flat)), shape=mask_dilate.shape)
# Convert it to array
arr = mtx.toarray().astype('float')

fig = px.imshow(image_seq_inpainted[50], color_continuous_scale='gray')
fig.add_trace(go.Contour(z=arr, contours_coloring='lines'))
plotly.io.show(fig)

plt.show()
