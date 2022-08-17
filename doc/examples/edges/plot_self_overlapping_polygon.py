"""
===================================
Subdivide self-overlapping polygons
===================================

This example shows how to divide a self-overlapping polygon into non
self-overlapping sub polygons using the method [1]_.
Each sub polygon can then be drawn or analyzed separately.

.. [1] Uddipan Mukherjee, "Self-overlapping curves: Analysis and applications,"
    Computer-Aided Design, 2014, 46, 227-232.
    :DOI:`10.1016/j.cad.2013.08.037`
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import draw, measure


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

rr, cc = draw.polygon(poly[:, 0], poly[:, 1], img.shape)
img[rr, cc, 1] = 1

# Subdivide the polygon into non self-overlapping sub polygons
sub_polys = measure.divide_selfoverlapping(poly)

print("Number of sub polygons:", len(sub_polys))

for sub_poly in sub_polys:
    rr, cc = draw.polygon(sub_poly[:, 0], sub_poly[:, 1], img.shape)
    img_subpolys[rr, cc, :] = np.random.rand(3)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(9, 4))

# Overlapped areas in self-overlapping polygons are drawn as holes.
ax1.imshow(img)
ax1.set_title("Overlaps in self-overlapping\n"
              "polygons appear as\n"
              "holes when drawn")

# Draw each sub polygon with a different color.
ax2.imshow(img_subpolys)
ax2.set_title("Each sub polygon,\n"
              "obtained after \n"
              "dividing the main\n"
              "polygon, drawn with\n"
              "different colors")

# Paint all sub polygon using the same color.
for sub_poly in sub_polys:
    rr, cc = draw.polygon(sub_poly[:, 0], sub_poly[:, 1], img.shape)
    img_subpolys[rr, cc, :] = [0, 1, 0]

ax3.imshow(img_subpolys)
ax3.set_title("All sub polygons\n"
              "drawn using\n"
              "the same color\n"
              "now without holes")

plt.show()
