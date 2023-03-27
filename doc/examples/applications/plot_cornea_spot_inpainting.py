"""
============================================================
Reconstruct dust-covered human cornea image using inpainting
============================================================

It is possible that dust gets accumulated on the reference mirror and causes
dark spots to appear on direct images. This example reproduces the steps taken
to perform OCT (Optical Coherence Tomography [1]_) dust removal in images.
This application was first discussed by Jules Scholler in [2]_.

.. [1] Vinay A. Shah M.D. (2015)
       `Optical Coherence Tomography <https://eyewiki.aao.org/Optical_Coherence_Tomography#:~:text=3%20Limitations-,Overview,at%20least%2010%2D15%20microns.>`_,
       American Academy of Ophthalmology.
.. [2] Jules Scholler (2019) "Image denoising using inpainting":
       `<https://www.jscholler.com/2019-02-28-remove-dots/>`_

"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import plotly.express as px


#################################################################################
# Load image
# ==========
# The dataset that we are using in this example is an image sequence showing
# the palisades of Vogt in a human cornea in vivo. The file has been
# acquired in TIFF format.

image_sequence = iio.imread('https://gitlab.com/mkcor/data/-/raw/70eb189f9b1c512fc8926891a2bdf96b67dcf441/in-vivo-cornea-spots.tif')

print(f'number of dimensions: {image_sequence.ndim}')
print(f'shape: {image_sequence.shape}')
print(f'dtype: {image_sequence.dtype}')

#################################################################################
# We visualize the image sequence by taking advantage of the *animation_feature*
# parameter in Plotly's *imshow* function. We set this feature to 0 to slice the
# image sequence along the temporal axis.

fig = px.imshow(image_sequence, animation_frame=0, binary_string=True,
                labels=dict(animation_frame="slice"),
                height=500, width=500,
                title="Animated Visualization")
fig.show()

#################################################################################
# In order to create a mask, we use the statistically average image over all the
# images in the TIFF file. This average frame is computed over time.

image_sequence_mean = np.sum(image_sequence, axis=0)
image_sequence_mean.shape

fig, ax = plt.subplots()
ax.set_title('image_sequence_mean')
plt.imshow(image_sequence_mean, cmap="gray")
