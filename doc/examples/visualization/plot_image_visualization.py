"""
==========================================================
Displaying images with some Python visualization libraries
==========================================================

How to visualize image arrays in 2D and in 3D with several Python visualization
libraries: Matplotlib, plotly, Mayavi.
"""
import numpy as np
from skimage import data
import matplotlib.pyplot as plt

img = data.astronaut()

plt.figure()
plt.imshow(img)
plt.show()

######################################################################
# Now with plotly
# ===============

import plotly.express as px

fig = px.imshow(img)
fig

######################################################################
# 3d data
# =======

from mayavi import mlab

img = data.binary_blobs(length=64, n_dim=3, blob_size_fraction=0.25)
img = img.astype(np.uint8)
mlab.figure()
mlab.contour3d(img, contours=[0.5])
