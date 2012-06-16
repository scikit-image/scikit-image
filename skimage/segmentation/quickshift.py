import numpy as np
from itertools import product, combinations_with_replacement

from IPython.core.debugger import Tracer
tracer = Tracer()


def quickshift(image, sigma=5, tau=10):
    # do smoothing beforehand?
    width, height = image.shape[:2]
    densities = np.zeros((width, height))
    w = 10

    # TODO: normalize density by number of considered points.
    # important  for the border!
    # compute densities
    for x, y in product(xrange(width), xrange(height)):
        current_pixel = np.hstack([image[x, y, :], x, y])
        for xx, yy in combinations_with_replacement(xrange(-w / 2, w / 2), 2):
            x_, y_ = x + xx, y + yy
            if 0 <= x_ < width and 0 <= y_ < height:
                other_pixel = np.hstack([image[x_, y_, :], x_, y_])
                dist = np.sum((current_pixel - other_pixel) ** 2)
                densities[x, y] += np.exp(-dist / sigma)

    # default parent to self:
    parent = np.arange(width * height).reshape(width, height)
    # find nearest node with higher density
    for x, y in product(xrange(width), xrange(height)):
        current_density = densities[x, y]
        current_pixel = np.hstack([image[x, y, :], x, y])
        closest = np.inf
        for xx, yy in combinations_with_replacement(xrange(-w / 2, w / 2), 2):
            x_, y_ = x + xx, y + yy
            if 0 <= x_ < width and 0 <= y_ < height:
                if densities[x_, y_] > current_density:
                    other_pixel = np.hstack([image[x_, y_, :], x_, y_])
                    dist = np.sum((current_pixel - other_pixel) ** 2)
                    if dist < closest:
                        closest = dist
                        parent[x, y] = x_ * width + y_
    flat = parent.ravel()
    old = np.zeros_like(flat)
    while (old != flat).any():
        old = flat
        flat = flat[flat]
    return flat.reshape(parent.shape)
