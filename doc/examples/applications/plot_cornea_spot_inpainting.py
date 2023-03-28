"""
============================================================
Reconstruct dust-covered human cornea image using inpainting
============================================================

Optical Coherence Tomography (OCT) [1]_ is used to provide eye doctors with an image
of the retina in the back of a patient's eye. It utilizes the concept of inferometry
to create a cross-sectional map of the retina that is accurate to within at least 10-15
microns. From its inception, OCT images have been acquired in a time domain fashion. It is
useful in the diagnosis of many retinal conditions, especially when the media is clear.

Quite commonly, dust gets accumulated on the reference mirror of the equipment and causes
dark spots to appear on images. This could reduce the accuracy of an optometrist's diagnosis.
In this example, we reproduce the steps taken to perform OCT dust removal in an image to
restore it to its original form. This application was first discussed by Jules Scholler
in [2]_.

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
# The dataset that we are using in this example is an image sequence showing the palisades of
# Vogt in a human cornea in vivo. The file has been acquired in TIFF format.

image_sequence = iio.imread('https://gitlab.com/mkcor/data/-/raw/70eb189f9b1c512fc8926891a2bdf96b67dcf441/in-vivo-cornea-spots.tif')

print(f'number of dimensions: {image_sequence.ndim}')
print(f'shape: {image_sequence.shape}')
print(f'dtype: {image_sequence.dtype}')

#################################################################################
# The dataset is a timeseries of 60 images, we visualize the image sequence by taking advantage
# of the *animation_feature* parameter in Plotly's *imshow* function. We set this feature to 0 to
# slice the image sequence along the temporal axis.

fig = px.imshow(image_sequence, animation_frame=0, binary_string=True,
                labels=dict(animation_frame="slice"),
                height=500, width=500,
                title="Animated Visualization")
fig.show()

###################################################################################
# To restore the dust-covered dark spots in the image sequence, we need to contrast these spots
# from the image background. This can be done by creating a thresholding mask that would be applied to all
# the frames (2D arrays) in the image sequence (3D array). We can say without doubt that the
# dark spots remain static through all the frames (or time points) in the sequence. Thus, we
# compute the 'time-averaged' image frame (along axis=0) to highlight these dark spots.

image_sequence_mean = np.mean(image_sequence, axis=0)
image_sequence_mean.shape

fig, ax = plt.subplots()
ax.set_title('image_sequence_mean')
ax.imshow(image_sequence_mean, cmap="gray")
