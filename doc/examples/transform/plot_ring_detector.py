"""
=============================
Ridge-directed ring detection
=============================

The ridge-directed Hough transform uses the local orientation of each ridge
pixel to vote only for circle centers along the ridge normal. This sparse
voting strategy avoids constructing the dense edge-based Hough transform for
every possible center. The detected circles are subsequently fitted to nearby
ridge pixels to obtain subpixel center and radius estimates.

Here, two blurred circular ridges illustrate the detector. ``sigma`` should
roughly match the ridge width. ``circle_threshold`` is the required number of
votes divided by radius; an ideal complete circle has a score close to
``2 * pi``.

The method was introduced in [1]_.

.. [1] E. Afik, "Robust and highly performant ring detection algorithm for
       3D particle tracking using 2D microscope imaging", Scientific Reports
       5, 13584 (2015). :doi:`10.1038/srep13584`
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage.draw import circle_perimeter
from skimage.transform import hough_ridge


image = np.zeros((160, 200), dtype=float)
for row, column, radius in ((55, 65, 28), (102, 137, 37)):
    rr, cc = circle_perimeter(row, column, radius, shape=image.shape)
    image[rr, cc] = 1
image = ndi.gaussian_filter(image, sigma=1.2)

rings = hough_ridge(
    image,
    (20, 45),
    sigma=1.2,
    vote_threshold=4,
    circle_threshold=1.5,
    ring_width=2,
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(image, cmap='gray')
for row, column, radius in rings:
    ax.add_patch(
        plt.Circle((column, row), radius, color='tab:red', fill=False, linewidth=2)
    )
ax.set_title(f'Ridge-directed detection: {len(rings)} rings')
ax.set_axis_off()
plt.show()
