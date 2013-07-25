import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from tex import growImage


__all__ = ['demo_syn']


def demo_syn():
    image = data.camera()[400:500, 250:350]
    paint_region = (slice(20, 25), slice(40, 50))
    image[paint_region] = 0

    mask = np.zeros_like(image, dtype=np.uint8)
    mask[paint_region] = 1

    return image, growImage(image, mask, 3)


# def demo_time_fill():
#     image = np.ones((15, 15), np.float)
#     fill_region = (slice(3, 12), slice(3, 12))
#     image[fill_region] = 0

#     mask = np.zeros_like(image, dtype=np.uint8)
#     mask[fill_region] = 1

#     return image, np.round(u, 1)


def start():

    image, painted = demo_syn()
    # image, painted = demo_time_fill()

    plt.gray()
    fig, (ax0, ax1) = plt.subplots(ncols=2)
    ax0.imshow(image)
    ax1.imshow(painted)
    plt.show()

start()
