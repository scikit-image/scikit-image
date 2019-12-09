"""
=========================
Measure region properties
=========================

This example shows how to measure properties of labelled image regions.

"""
import math
import matplotlib.pyplot as plt
import numpy as np

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate


image = np.zeros((600, 600))

rr, cc = ellipse(300, 350, 100, 220)
image[rr, cc] = 1

image = rotate(image, angle=15, order=0)

label_img = label(image)
props = (
    'centroid', 'orientation',
    'major_axis_length', 'minor_axis_length',
    'bbox'
)

table = regionprops_table(label_img, image, properties=props)

fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)

for r in range(len(table['centroid'])):
    y0, x0 = table['centroid'][r]
    orientation = table['orientation'][r]
    x1 = x0 + math.cos(orientation) / 2 * table['major_axis_length'][r]
    y1 = y0 - math.sin(orientation) / 2 * table['major_axis_length'][r]
    x2 = x0 - math.sin(orientation) / 2 * table['minor_axis_length'][r]
    y2 = y0 - math.cos(orientation) / 2 * table['minor_axis_length'][r]

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

    minr = table['bbox-0'][r]
    minc = table['bbox-1'][r]
    maxr = table['bbox-2'][r]
    maxc = table['bbox-3'][r]
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)

ax.axis((0, 600, 600, 0))
plt.show()
