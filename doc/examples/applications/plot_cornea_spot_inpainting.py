"""
============================================
Restore spotted cornea image with inpainting
============================================

Optical coherence tomography (OCT) is a non-invasive imaging technique used by
ophthalmologists to take pictures of the back of a patient's eye [1]_.
When performing OCT,
dust may stick to the reference mirror of the equipment, causing dark spots to
appear on the images. The problem is that these dirt spots cover areas of
in-vivo tissue, hence hiding data of interest. Our goal here is to restore
(reconstruct) the hidden areas based on the pixels near their boundaries.

This tutorial is adapted from an application shared by Jules Scholler [2]_.
The images were acquired by Viacheslav Mazlin (see
:func:`skimage.data.palisades_of_vogt`).

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

import skimage as ski


#####################################################################
# The dataset we are using here is an image sequence (a movie!) of
# human in-vivo tissue. Specifically, it shows the *palisades of Vogt* of a
# given cornea sample.

#####################################################################
# Load image data
# ===============

image_seq = ski.data.palisades_of_vogt()

print(f'number of dimensions: {image_seq.ndim}')
print(f'shape: {image_seq.shape}')
print(f'dtype: {image_seq.dtype}')

#####################################################################
# The dataset is an image stack with 60 frames (time points) and 2 spatial
# dimensions. Let us visualize 10 frames by sampling every six time points:
# We can see some changes in illumination.
# We take advantage of the ``animation_frame`` parameter in
# Plotly's ``imshow`` function. As a side note, when the
# ``binary_string`` parameter is set to ``True``, the image is
# represented as grayscale.

fig = px.imshow(
    image_seq[::6, :, :],
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': '6-step time point'},
    title='Sample of in-vivo human cornea',
)
fig.update_layout(autosize=False, minreducedwidth=250, minreducedheight=250)
plotly.io.show(fig)

#####################################################################
# Aggregate over time
# ===================
# First, we want to detect those dirt spots where the data are lost. In
# technical terms, we want to *segment* the dirt spots (for
# all frames in the sequence). Unlike the actual data (signal), the dirt spots
# do not move from one frame to the next; they are still. Therefore, we begin
# by computing a time aggregate of the image sequence. We shall use the median
# image to segment the dirt spots, the latter then standing out
# with respect to the background (blurred signal).
# Complementarily, to get a feel for the (moving) data, let us compute the
# variance.

image_med = np.median(image_seq, axis=0)
image_var = np.var(image_seq, axis=0)

assert image_var.shape == image_med.shape

print(f'shape: {image_med.shape}')

fig, ax = plt.subplots(ncols=2, figsize=(12, 6))

ax[0].imshow(image_med, cmap='gray')
ax[0].set_title('Image median over time')
ax[1].imshow(image_var, cmap='gray')
ax[1].set_title('Image variance over time')

fig.tight_layout()

#####################################################################
# Use local thresholding
# ======================
# To segment the dirt spots, we use thresholding. The images we are working
# with are unevenly illuminated, which causes spatial variations in the
# (absolute) intensities of the foreground and the background. For example,
# the average background intensity in one region may be different in another
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

thresh_1 = ski.filters.threshold_local(image_med, block_size=21, offset=15)
thresh_2 = ski.filters.threshold_local(image_med, block_size=43, offset=15)

mask_1 = image_med < thresh_1
mask_2 = image_med < thresh_2

#####################################################################
# Let us define a convenience function to display two plots side by side, so
# it is easier for us to compare them:


def plot_comparison(plot1, plot2, title1, title2):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)
    ax1.imshow(plot1, cmap='gray')
    ax1.set_title(title1)
    ax2.imshow(plot2, cmap='gray')
    ax2.set_title(title2)


plot_comparison(mask_1, mask_2, "block_size = 21", "block_size = 43")

#####################################################################
# The "dirt spots" appear to be more distinct in the second mask, i.e., the
# one resulting from using the larger ``block_size`` value.
# We noticed that increasing the value of the offset parameter from
# its default zero value would yield a more uniform background,
# letting the objects of interest stand out more visibly. Note that
# toggling parameter values can give us a deeper
# understanding of the method being used, which can typically move us
# closer to the desired results.

thresh_0 = ski.filters.threshold_local(image_med, block_size=43)

mask_0 = image_med < thresh_0

plot_comparison(mask_0, mask_2, "No offset", "Offset = 15")

#####################################################################
# Remove fine-grained features
# ============================
# We use morphological filters to sharpen the mask and focus on the dirt
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

footprint = ski.morphology.diamond(3)
mask_open = ski.morphology.opening(mask_2, footprint)
plot_comparison(mask_2, mask_open, "mask before", "after opening")

#####################################################################
# Since "opening" an image starts with an erosion operation, bright regions
# which are smaller than the structuring element have been removed.
# When applying an opening filter, tweaking the footprint parameter lets us
# control how fine-grained the removed features are. For example, if we used
# ``footprint = ski.morphology.diamond(1)`` in the above, we could see that
# only smaller features would be filtered out, hence retaining more spots in
# the mask. Conversely, if we used a disk-shaped footprint of same radius,
# i.e., ``footprint = ski.morphology.disk(3)``, more of the fine-grained
# features would be filtered out, since the disk's area is larger than the
# diamond's.

#####################################################################
# Next, we can make the detected areas wider by applying a dilation filter:

mask_dilate = ski.morphology.dilation(mask_open, footprint)
plot_comparison(mask_open, mask_dilate, "Before", "After dilation")

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
    image_seq_inpainted[i] = ski.restoration.inpaint_biharmonic(
        image_seq[i], mask_dilate
    )

#####################################################################
# Let us visualize one restored image, where the dirt spots have been
# inpainted. First, we find the contours of the dirt spots (well, of the mask)
# so we can draw them on top of the restored image:

contours = ski.measure.find_contours(mask_dilate)

# Gather all (row, column) coordinates of the contours
x = []
y = []
for contour in contours:
    x.append(contour[:, 0])
    y.append(contour[:, 1])
# Note that the following one-liner is equivalent to the above:
# x, y = zip(*((contour[:, 0], contour[:, 1]) for contour in contours))

# Flatten the coordinates
x_flat = np.concatenate(x).ravel().round().astype(int)
y_flat = np.concatenate(y).ravel().round().astype(int)
# Create mask of these contours
contour_mask = np.zeros(mask_dilate.shape, dtype=bool)
contour_mask[x_flat, y_flat] = 1
# Pick one frame
sample_result = image_seq_inpainted[12]
# Normalize it (so intensity values range [0, 1])
sample_result /= sample_result.max()

#####################################################################
# We use function ``label2rgb`` from the ``color`` module to overlay the
# restored image with the segmented spots, using transparency (alpha
# parameter).

color_contours = ski.color.label2rgb(
    contour_mask, image=sample_result, alpha=0.4, bg_color=(1, 1, 1)
)

fig, ax = plt.subplots(figsize=(6, 6))

ax.imshow(color_contours)
ax.set_title('Segmented spots over restored image')

fig.tight_layout()

#####################################################################
# Note that the dirt spot located at (x, y) ~ (719, 1237) stands out; ideally,
# it should have been segmented and inpainted. We can see that we 'lost' it to
# the opening processing step, when removing fine-grained features.

plt.show()
