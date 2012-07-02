import numpy as np
cimport numpy as np
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
    cdef np.ndarray[dtype=np.float_t, ndim=2] means = np.dstack([means_y, means_x, means_color]).reshape(-1, 5)
    n_means = means.shape[0]
    # we do the scaling of ratio in the same way as in the SLIC paper
    # so the values have the same meaning
    ratio = (ratio / float(step)) ** 2
    print(ratio)
    cdef np.ndarray[dtype=np.float_t, ndim=3] image_yx = np.dstack([grid_y, grid_x, image / ratio])
    cdef int i, k, x, y, x_min, x_max, y_min, y_max
    cdef float dist_mean

    cdef np.ndarray[dtype=np.int_t, ndim=2] nearest_mean = np.zeros((height, width), dtype=np.int)
    cdef np.ndarray[dtype=np.float_t, ndim=2] distance = np.ones((height, width), dtype=np.float) * np.inf
    for i in xrange(max_iter):
        print("iteration %d" % i)
        nearest_mean_old = nearest_mean.copy()
        # assign pixels to means
        for k in xrange(n_means):
            # compute windows:
            y_min = int(max(means[k, 0] - 2 * step, 0))
            y_max = int(min(means[k, 0] + 2 * step, height))
            x_min = int(max(means[k, 1] - 2 * step, 0))
            x_max = int(min(means[k, 1] + 2 * step, height))
            for x in xrange(x_min, x_max):
                for y in xrange(y_min, y_max):
                    dist_mean = 0
                    for c in range(5):
                        dist_mean += (image_yx[y, x, c] - means[k, c]) ** 2
                    if distance[y, x] > dist_mean:
                        nearest_mean[y, x] = k
                        distance[y, x] = dist_mean
        if (nearest_mean == nearest_mean_old).all():
            break
        # recompute means:
        means_list = [np.bincount(nearest_mean.ravel(), image_yx[:, :, j].ravel())
                for j in xrange(5)]
        in_mean = np.bincount(nearest_mean.ravel())
        means = (np.vstack(means_list) / in_mean).T
    return nearest_mean
