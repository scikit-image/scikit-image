"""
===================================
Separate self-overlapping polygons
===================================

This example shows how to separate a self-overlapping polygon into non
self-overlapping sub-polygons using the method proposed in [1]_.
Each sub-polygon can then be drawn or analyzed separately.

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
img_subpolys_exp = np.zeros((1536, 1536, 3), dtype=np.double)

rr, cc = draw.polygon(poly[:, 0], poly[:, 1], img.shape)
img[rr, cc, 1] = 1

# Subdivide the polygon into non self-overlapping sub polygons
sub_polys = measure.separate_selfoverlapping_polygon(poly)

print("Number of sub polygons:", len(sub_polys))

for s, sub_poly in enumerate(sub_polys):
    rr, cc = draw.polygon(sub_poly[:, 0], sub_poly[:, 1], img.shape)
    poly_col = np.random.rand(3) * 0.75 + 0.25
    img_subpolys[rr, cc, :] = poly_col
    rr = rr - np.mean(rr).astype(np.int64) + 256 + (s // 3) * 512
    cc = cc - np.mean(cc).astype(np.int64) + 256 + (s % 3) * 512
    img_subpolys_exp[rr, cc, :] = poly_col

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))

# Overlapped areas in self-overlapping polygons are drawn as holes.
ax1.imshow(img)
ax1.plot(poly[:, 1], poly[:, 0], 'b-')
ax1.plot([poly[-1, 1], poly[0, 1]], [poly[-1, 0], poly[0, 0]], 'b-')
ax1.set_title("Overlaps in self-overlapping polygons\n"
              "appear as holes when drawn")
ax1.axis('off')

# Show each sub polygon.
ax2.imshow(img_subpolys_exp)
ax2.set_title("The sub polygons obtained after\n"
              "dividing the main polygon")
ax2.axis('off')

# Draw each sub polygon with a different color.
ax3.imshow(img_subpolys)
ax3.set_title("Each sub polygon drawn\n"
              "with different color")
ax3.plot(poly[:, 1], poly[:, 0], 'b-')
ax3.plot([poly[-1, 1], poly[0, 1]], [poly[-1, 0], poly[0, 0]], 'b-')
ax3.axis('off')

# Paint all sub polygon using the same color.
for sub_poly in sub_polys:
    rr, cc = draw.polygon(sub_poly[:, 0], sub_poly[:, 1], img.shape)
    ax3.plot(poly[:, 1], poly[:, 0], 'b:')
    ax3.plot([sub_poly[-1, 1], sub_poly[0, 1]],
             [sub_poly[-1, 0], sub_poly[0, 0]],
             'b:')
    ax4.plot(poly[:, 1], poly[:, 0], 'b:')
    ax4.plot([sub_poly[-1, 1], sub_poly[0, 1]],
             [sub_poly[-1, 0], sub_poly[0, 0]],
             'b:')
    img_subpolys[rr, cc, :] = [0, 1, 0]

ax4.imshow(img_subpolys)
ax4.set_title("All sub polygons drawn using\n"
              "the same color now without holes")
ax4.axis('off')

plt.show()
