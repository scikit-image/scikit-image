import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filter.inpaint_texture import inpaint_efros
# plt.gray()


__all__ = ['demo_syn']


def demo_syn():
    image = data.camera()[400:500, 250:350]
    paint_region = (slice(20, 45), slice(40, 60))

    mask = np.zeros_like(image, dtype=np.uint8)
    image[paint_region] = 0
    mask[paint_region] = 1

    return image, inpaint_efros(image, mask, 9)


def start():

    image, painted = demo_syn()

    fig, (ax0, ax1) = plt.subplots(ncols=2)
    ax0.imshow(image)
    ax1.imshow(painted)
    plt.show()

start()
