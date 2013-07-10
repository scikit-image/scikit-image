import numpy as np
plt.gray()

import _heap
import fmm


def inpaint(image, mask, epsilon=3):
    image = image.copy()

    flag, u, heap = _heap.initialise(mask)

    painted = fmm.fast_marching_method(image, flag, u, heap, epsilon=epsilon)
    return painted


def demo_inpaint():
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


def demo_time_fill():
    image = np.ones((10, 10))
    fill_region = (slice(4, -4), slice(4, -4))
    image[fill_region] = 0

    mask = np.zeros_like(image, dtype=int)
    mask[fill_region] = 1

    flag, u, heap = _heap.initialise(mask)
    time_map = fmm.fast_marching_method(image, flag, u, heap,
                                        _run_inpaint=False, epsilon=3)
    print np.round(time_map, 1)

    fig, ax = plt.subplots()
    ax.imshow(time_map)
    plt.show()


# demo_inpaint()
demo_time_fill()
