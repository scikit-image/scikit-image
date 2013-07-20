import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# import _heap
# import fmm

from c_inpaint import inpaint, initialise, fast_marching_method

__all__ = ['demo_inpaint', 'demo_time_fill']


def demo_inpaint(image, mask):
    print np.sum(mask)
    # painted = fmm.inpaint(image, mask)
    painted = inpaint(image, mask)

    fig, (ax0, ax1) = plt.subplots(ncols=2)
    plt.gray()
    ax0.imshow(image)
    ax1.imshow(painted)
    plt.show()
    return painted


def demo_time_fill():
    image = np.ones((100, 100))
    fill_region = (slice(30, 40), slice(50, 70))

    image[fill_region] = 0

    mask = np.zeros_like(image, dtype=np.uint8)
    mask[fill_region] = 1
