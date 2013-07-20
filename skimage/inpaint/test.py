import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import _heap
import fmm


__all__ = ['demo_inpaint', 'demo_time_fill']


def demo_inpaint(image, mask):
    return fmm.inpaint(image, mask)


def demo_time_fill():
    image = np.ones((15, 15), np.uint8)
    fill_region = (slice(3, 12), slice(3, 12))
    image[fill_region] = 0

    mask = np.zeros_like(image, dtype=np.uint8)
    mask[fill_region] = 1

    flag, u, heap = _heap.initialise(mask)
    time_map = fmm.fast_marching_method(image, flag, u, heap,
                                        _run_inpaint=False, epsilon=5)

    return image, np.round(time_map, 1)


def start():
    image = data.camera()
    paint_region = (slice(20, 30), slice(40, 70))
    image[paint_region] = 0

    mask = np.zeros_like(image, dtype=np.uint8)
    mask[paint_region] = 1

    painted = demo_inpaint(image, mask)
    fig, (ax0, ax1) = plt.subplots(ncols=2)
    plt.gray()
    ax0.imshow(image)
    ax1.imshow(painted)
    plt.show()

start()
