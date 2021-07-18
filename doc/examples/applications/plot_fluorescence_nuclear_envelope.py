"""
======================================================
Measure fluorescence intensity at the nuclear envelope
======================================================

online tutorial [1]_ based on publication [2]_

.. [1] https://doi.org/10.1007/978-3-030-22386-1_2
.. [2] Boni A, Politi AZ, Strnad P, Xiang W, Hossain MJ, Ellenberg J (2015)
       "Live imaging and modeling of inner nuclear membrane targeting reveals
       its molecular requirements in mammalian cells" J Cell Biol
       209(5):705–720. ISSN: 0021-9525.
       :DOI:`10.1083/jcb.201409133`

"""

import imageio
import matplotlib.pyplot as plt
import plotly.io
import plotly.express as px
from scipy import ndimage as ndi

from skimage import (
    filters, measure, morphology, segmentation
)


#####################################################################
# We start with a single cell/nucleus to construct the workflow.

one = imageio.volread('http://cmci.embl.de/sampleimages/NPCsingleNucleus.tif')

print(f'shape: {one.shape}')

#####################################################################
# The dataset is a 2D image stack with 15 frames (time points) and 2 channels.

fig = px.imshow(
    one,
    facet_col=1,
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': 'time point', 'facet_col': 'channel'}
)
plotly.io.show(fig)

#####################################################################
# To begin with, let us consider the first channel of the first image.

ch0t0 = one[0, 0, :, :]

smooth = filters.gaussian(ch0t0, sigma=2)

thresh = smooth > 0.1

fill = ndi.binary_fill_holes(thresh)

fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

ax[0].imshow(ch0t0, cmap=plt.cm.gray)
ax[0].set_title('0) Raw')

ax[1].imshow(smooth, cmap=plt.cm.gray)
ax[1].set_title('1) Smooth out')

ax[2].imshow(thresh, cmap=plt.cm.gray)
ax[2].set_title('2) Threshold')

ax[3].imshow(fill, cmap=plt.cm.gray)
ax[3].set_title('3) Fill in')

for a in ax:
    a.axis('off')

plt.show()
