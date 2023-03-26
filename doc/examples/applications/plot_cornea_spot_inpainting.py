"""
===============================================
Reconstruct dust-covered image using inpainting
===============================================

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
# The dataset that we are using in this example is an image sequence showing
# the palisades of Vogt in vivo in a human cornea. The file has been
# acquired in TIFF format.

raw = iio.imread('data/raw.tiff')

print(f'shape: {raw.shape}')

#################################################################################
# We visualise the image sequence by taking advantage of  the *animation_feature*
# parameter in Plotly's *imshow* function. We set this feature to 0 to slice the
# image sequence along the temporal axis.

fig = px.imshow(raw, animation_frame=0, binary_string=True,
                labels=dict(animation_frame="slice"),
                height=500, width=500,
                title="Animated Visualisation")
fig.show()

#################################################################################
# In order to create a mask, we use the statistically average image over all the
# images in the TIFF file. This average frame is computed over time.

raw_mean = np.sum(raw, axis=0)
raw_mean.shape

fig, ax = plt.subplots()
ax.set_title('raw_mean')
plt.imshow(raw_mean, cmap="gray")
