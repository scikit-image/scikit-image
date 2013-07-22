import numpy as np
import matplotlib.pyplot as plt
from skimage import data

import inpaint
import _inpaint

# import _heap
# import fmm

from c_inpaint import inpaint, initialise, fast_marching_method

__all__ = ['demo_inpaint', 'demo_time_fill']


def demo_inpaint(image, mask):
    return inpaint.inpaint(image, mask)


def demo_time_fill():
    image = np.ones((15, 15), np.uint8)
    fill_region = (slice(3, 12), slice(3, 12))
    image[fill_region] = 0

    mask = np.zeros_like(image, dtype=np.uint8)
    mask[fill_region] = 1

    flag, u, heap = inpaint.initialise(mask)
    time_map = _inpaint.fast_marching_method(image, flag, u, heap,
                                             _run_inpaint=False, epsilon=5)

    return image, np.round(time_map, 1)


def start():
    image = data.camera()
    paint_region = (slice(120, 130), slice(440, 470))
    paint_region1 = (slice(220, 230), slice(440, 470))
    image[paint_region] = 0
    image[paint_region1] = 0

    mask = np.zeros_like(image, dtype=np.uint8)
    mask[paint_region] = 1
    mask[paint_region1] = 1

    painted = demo_inpaint(image, mask)

    fig, (ax0, ax1) = plt.subplots(ncols=2)
    plt.gray()
    ax0.imshow(image)
    ax1.imshow(painted)
    plt.show()
    return painted
