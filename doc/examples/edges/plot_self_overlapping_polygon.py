"""
==================================
Subdivide self-overlapping polygons
===================================

This example shows how to divide a self-overlapping polygon into non
self-overlapping sub-polygons. Each sub-polygon can then be drawn or analyzed
separately.

"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import divide_selfoverlapping
from skimage.draw import polygon


poly = np.array([[509, 512],
                 [5, 302],
                 [290, 58],
                 [181, 309],
                 [439, 378],
                 [438, 100],
                 [173, 213],
                 [262, 489],
                 [0, 242],
                 [512, 0]])

img = np.zeros((512, 512, 3), dtype=np.double)
img_subpolys = np.zeros((512, 512, 3), dtype=np.double)

rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
img[rr, cc, 1] = 1

# Subdivide the polygon into non self-overlapping sub-polygons
sub_polys = divide_selfoverlapping(poly)

print("Number of sub-polygons:", len(sub_polys))

for sub_poly in sub_polys:
    rr, cc = polygon(sub_poly[:, 0], sub_poly[:, 1], img.shape)
    img_subpolys[rr, cc, :] = np.random.rand(3)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(9, 4))

# Overlapped areas are drawn as holes.
ax1.imshow(img)

# Draw each sub-polygon with a different color.
ax2.imshow(img_subpolys)

# Paint all sub-polygon using the same color.
for sub_poly in sub_polys:
    rr, cc = polygon(sub_poly[:, 0], sub_poly[:, 1], img.shape)
    img_subpolys[rr, cc, :] = [0, 1, 0]

ax3.imshow(img_subpolys)

plt.show()
