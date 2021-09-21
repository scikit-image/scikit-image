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
import numpy as np
import plotly.io
import plotly.express as px
from scipy import ndimage as ndi

from skimage import (
    filters, measure, morphology, segmentation
)


#####################################################################
# We start with a single cell/nucleus to construct the workflow.

dat = imageio.volread('http://cmci.embl.de/sampleimages/NPCsingleNucleus.tif')

print(f'shape: {dat.shape}')

#####################################################################
# The dataset is a 2D image stack with 15 frames (time points) and 2 channels.

fig = px.imshow(
    dat,
    facet_col=1,
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': 'time point', 'facet_col': 'channel'}
)
plotly.io.show(fig)

#####################################################################
# To begin with, let us consider the first channel of the first image.

ch0t0 = dat[0, 0, :, :]

#####################################################################
# Segment the nucleus rim
# =======================

smooth = filters.gaussian(ch0t0, sigma=2)

thresh = smooth > 0.1

fill = ndi.binary_fill_holes(thresh)

#####################################################################
# Although it is not strictly necessary, let us discard the other nucleus part
# visible in the bottom right-hand corner.

label = measure.label(fill)
regions = measure.regionprops_table(label, properties=('label', 'area'))
label_small = regions['label'][np.argmin(regions['area'])]
label[label == label_small] = 0  # background value

expand = segmentation.expand_labels(label, distance=4)

erode = morphology.erosion(label)

fig, ax = plt.subplots(2, 4, figsize=(12, 6), sharey=True)

ax[0, 0].imshow(ch0t0, cmap=plt.cm.gray)
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

ax[1, 3].imshow(expand - erode, cmap=plt.cm.gray)
ax[1, 3].set_title('f) Nucleus Rim')

for a in ax.ravel():
    a.axis('off')

plt.show()
