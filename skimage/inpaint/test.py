import numpy as np
import matplotlib.pyplot as plt
from skimage import data

import _heap
import fmm


def inpaint(image, mask):
    # Generate `flag` and `u` matrices
    flag = _heap.init_flag(mask)
    u = _heap.init_u(flag)

    # Initialize the heap array
    heap = []
    _heap.generate_heap(heap, flag, u)

    painted = fmm.fast_marching_method(image, flag, u, heap, negate=False,
                                       epsilon=epsilon)
    return painted


image = data.camera()[80:180, 200:300]
paint_region = (slice(45, 55), slice(20, 80))
image[paint_region] = 0

mask = np.zeros_like(image, dtype=np.uint8)
mask[paint_region] = 1

epsilon = 3
con = image.copy()

output = inpaint(image, mask)


fig, (ax0, ax1) = plt.subplots(ncols=2)
plt.gray()
ax0.imshow(con)
ax1.imshow(output)
plt.show()
