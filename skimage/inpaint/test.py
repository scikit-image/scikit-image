import numpy as np
import matplotlib.pyplot as plt
from skimage import data

import _heap
import fmm


def inpaint(image, mask, epsilon=3):
    image = image.copy()

    flag, u, heap = _heap.initialise(mask)

    painted = fmm.fast_marching_method(image, flag, u, heap, epsilon=epsilon)
    return painted

image = data.camera()[80:180, 200:300]
paint_region = (slice(35, 45), slice(80, 95))
image[paint_region] = 0

mask = np.zeros_like(image, dtype=np.uint8)
mask[paint_region] = 1

painted = inpaint(image, mask)

fig, (ax0, ax1) = plt.subplots(ncols=2)
plt.gray()
ax0.imshow(image)
ax1.imshow(painted)
plt.show()
