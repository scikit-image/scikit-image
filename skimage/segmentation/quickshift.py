import numpy as np
from itertools import product


def quickshift(image, sigma=5, tau=10):
    """Computes quickshift clustering in RGB-(x,y) space.

    Parameters
    ----------
    image: ndarray, [width, height, channels]
        Input image
    sigma: float
        Width of Gaussian kernel used in smoothing the
        sample density. Higher means less clusters.
    tau: float
        Cut-off point for data distances.
        Higher means less clusters.

    Returns
    -------
    segment_mask: ndarray, [width, height]
        Integer mask indicating segment labels.
    """

    # We compute the distances twice since otherwise
    # we might get crazy memory overhead (width * height * windowsize**2)

    # TODO do smoothing beforehand?
    # TODO manage borders somehow?

    # window size for neighboring pixels to consider
    if sigma < 1:
        raise ValueError("Sigma should be >= 1")
    w = int(2 * sigma)

    width, height = image.shape[:2]
    densities = np.zeros((width, height))

    # compute densities
    for x, y in product(xrange(width), xrange(height)):
        current_pixel = np.hstack([image[x, y, :], x, y])
        for xx, yy in product(xrange(-w / 2, w / 2 + 1), repeat=2):
            x_, y_ = x + xx, y + yy
            if 0 <= x_ < width and 0 <= y_ < height:
                other_pixel = np.hstack([image[x_, y_, :], x_, y_])
                dist = np.sum((current_pixel - other_pixel) ** 2)
                densities[x, y] += np.exp(-dist / sigma)

    # this will break ties that otherwise would give us headache
    densities += np.random.normal(scale=0.00001, size=densities.shape)
    # default parent to self:
    parent = np.arange(width * height).reshape(width, height)
    dist_parent = np.zeros((width, height))
    # find nearest node with higher density
    for x, y in product(xrange(width), xrange(height)):
        current_density = densities[x, y]
        current_pixel = np.hstack([image[x, y, :], x, y])
        closest = np.inf
        for xx, yy in product(xrange(-w / 2, w / 2 + 1), repeat=2):
            x_, y_ = x + xx, y + yy
            if 0 <= x_ < width and 0 <= y_ < height:
                if densities[x_, y_] > current_density:
                    other_pixel = np.hstack([image[x_, y_, :], x_, y_])
                    dist = np.sum((current_pixel - other_pixel) ** 2)
                    if dist < closest:
                        closest = dist
                        parent[x, y] = x_ * width + y_
        dist_parent[x, y] = closest

    dist_parent = dist_parent.ravel()
    flat = parent.ravel()
    flat[dist_parent > tau] = np.arange(width * height)[dist_parent > tau]
    old = np.zeros_like(flat)
    while (old != flat).any():
        old = flat
        flat = flat[flat]
    return flat.reshape(parent.shape)
