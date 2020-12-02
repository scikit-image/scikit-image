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
import plotly

fig = px.imshow(img)
plotly.io.show(fig)

