"""
==========================================================
Displaying images with some Python visualization libraries
==========================================================

How to visualize image arrays in 2D and in 3D with several Python visualization
libraries: Matplotlib, plotly, Mayavi.
"""
import numpy as np
from skimage import data, filters, measure
from skimage import img_as_ubyte

import plotly.graph_objects as go
import plotly

img = data.cells3d()[:, 1]
img = img_as_ubyte(img)
img = filters.median(img, np.ones((3, 3, 3)))

threshold = filters.threshold_otsu(img)
mask = img > threshold
verts, faces, _, _ = measure.marching_cubes(img, threshold, step_size=2)
x, y, z = verts.T
i, j, k = faces.T
fig = go.Figure()
fig.add_trace(go.Mesh3d(x=z, y=y, z=x, opacity=0.2, i=k, j=j, k=i))


plotly.io.show(fig)

