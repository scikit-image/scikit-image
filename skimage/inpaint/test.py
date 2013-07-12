import numpy as np
import matplotlib.pyplot as plt
from skimage import data

import _heap
import fmm


__all__ = ['demo_inpaint', 'demo_time_fill']


def demo_inpaint(image, mask):
    painted = fmm.inpaint(image, mask)

    fig, (ax0, ax1) = plt.subplots(ncols=2)
    plt.gray()
    ax0.imshow(image)
    ax1.imshow(painted)
    plt.show()
    # return painted


def demo_time_fill():
    image = np.ones((100, 100))
    fill_region = (slice(30, 40), slice(50, 70))
    image[fill_region] = 0

    mask = np.zeros_like(image, dtype=int)
    mask[fill_region] = 1

    flag, u, heap = _heap.initialise(mask)
    time_map = fmm.fast_marching_method(image, flag, u, heap,
                                        _run_inpaint=False, epsilon=3)
    print np.round(time_map[25:45, 45:75], 1)

    fig, (ax0, ax1) = plt.subplots(ncols=2)
    ax0.imshow(mask[20:50, 40:80])
    ax1.imshow(time_map[20:50, 40:80])
    plt.show()


image = data.camera()[80:180, 200:300]
paint_region = (slice(55, 65), slice(85, 100))
image[paint_region] = 0

mask = np.zeros_like(image, dtype=np.uint8)
mask[paint_region] = 1
demo_inpaint(image, mask)
# demo_time_fill()
