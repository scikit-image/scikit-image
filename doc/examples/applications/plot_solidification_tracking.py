"""
========================================
Track solidification of a metallic alloy
========================================

In this example, we identify and track the solid-liquid (S-L) interface in a
metallic alloy sample undergoing solidification. The image sequence was
obtained by Gus Becker using synchrotron x-radiography at the Advanced Photon
Source (APS) of Argonne National Laboratory (ANL). This analysis is taken from
[source with DOI?], also presented in a conference talk [1]_.

.. [1] Corvellec M. and Becker C. G. (2021, May 17-18)
       "Quantifying solidification of metallic alloys with scikit-image"
       [Conference presentation]. BIDS ImageXD 2021 (Image Analysis Across
       Domains). Virtual participation.
       https://www.youtube.com/watch?v=cB1HTgmWTd8
"""

import numpy as np
import pandas as pd
import plotly.io
import plotly.express as px

from skimage import measure, segmentation
from skimage.data import nickel_solidification

image_sequence = nickel_solidification()

y1 = 0
y2 = 180
x1 = 100
x2 = 330

image_sequence = image_sequence[:, y1:y2, x1:x2]

print(f'shape: {image_sequence.shape}')

#####################################################################
# The dataset is a 2D image stack with 11 frames (time points).

fig = px.imshow(
    image_sequence,
    animation_frame=0,
    binary_string=True,
    labels={'animation_frame': 'time point'}
)
plotly.io.show(fig)
