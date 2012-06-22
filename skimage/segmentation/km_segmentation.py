import numpy as np
from scipy import ndimage
from ..util import img_as_float


def km_segmentation(image, n_segments=100, ratio=10., max_iter=100, sigma=1.0):
    image = ndimage.gaussian_filter(img_as_float(image), sigma)
    # initialize on grid:
    height, width = image.shape[:2]
    # approximate grid size for desired n_segments
    step = np.sqrt(height * width / n_segments)
    grid_y, grid_x = np.mgrid[:height, :width]
    means_y = grid_y[::step, ::step]
    means_x = grid_x[::step, ::step]

    means_color = image[means_y, means_x, :]
    means = np.dstack([means_y, means_x, means_color]).reshape(-1, 5)
    # we do the scaling of ratio in the same way as in the SLIC paper
    # so the values have the same meaning
    ratio = (ratio / float(step)) ** 2
    print(ratio)
    image = np.dstack([grid_y, grid_x, image / ratio])

    nearest_mean = np.zeros((height, width), dtype=np.int)
    distance = np.ones((height, width), dtype=np.float) * np.inf
    for i in xrange(max_iter):
        print("iteration %d" % i)
        nearest_mean_old = nearest_mean.copy()
        # assign pixels to means
        for k, mean in enumerate(means):
            # compute windows:
            y_min = int(max(mean[0] - 2 * step, 0))
            y_max = int(min(mean[0] + 2 * step, height))
            x_min = int(max(mean[1] - 2 * step, 0))
            x_max = int(min(mean[1] + 2 * step, height))
            search_window = image[y_min:y_max + 1, x_min:x_max + 1]
            dist_mean = np.sum((search_window - mean) ** 2, axis=2)
            assign = distance[y_min:y_max + 1, x_min:x_max + 1] > dist_mean
            nearest_mean[y_min:y_max + 1, x_min:x_max + 1][assign] = k
            distance[y_min:y_max + 1, x_min:x_max + 1][assign] = \
                    dist_mean[assign]
        if (nearest_mean == nearest_mean_old).all():
            break
        # recompute means:
        means = [np.bincount(nearest_mean.ravel(), image[:, :, j].ravel())
                for j in xrange(5)]
        in_mean = np.bincount(nearest_mean.ravel())
        means = (np.vstack(means) / in_mean).T
    return nearest_mean
