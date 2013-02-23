#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Author: Francois Boulogne
# License: GPL

"""
========================
Circular Hough Transform
========================



"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data, filter
from skimage.transform import hough_circle
from skimage.feature import peak_local_max

# Load picture and detect edges
image = data.coins()[0:95, 70:370]
edges = filter.canny(filter.sobel(image), sigma=2.8)


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(image, cmap=plt.cm.gray)

# Detect two radii
radii = np.array([21, 25])
hough_res = hough_circle(edges, radii)

for radius, h in zip(radii, hough_res):
    # For each radius, keep two circles
    maxima = peak_local_max(h, num_peaks=2)
    for maximum in maxima:
        center_x, center_y = maximum - radii.max()
        circ = mpatches.Circle((center_y, center_x), radius,
                               fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(circ)

plt.show()
