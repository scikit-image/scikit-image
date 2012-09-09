"""
====================
Build image pyramids
====================

The `build_gauassian_pyramid` function takes an image and yields successive
images shrunk by a constant scale factor. Image pyramids are often used, e.g.,
to implement algorithms for denoising, texture discrimination, and scale-
invariant detection.
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import img_as_float
from skimage.transform import build_gaussian_pyramid


image = data.lena()
rows, cols, dim = image.shape
pyramid = tuple(build_gaussian_pyramid(image, downscale=2))

display = np.zeros((rows, cols + cols / 2, 3), dtype=np.double)

display[:rows, :cols, :] = pyramid[0]

i_row = 0
for p in pyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    display[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

plt.imshow(display)
plt.show()
