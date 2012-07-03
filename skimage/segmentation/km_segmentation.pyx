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
    cdef np.float_t* current_mean
    cdef np.float_t* mean_entry
    n_means = means.shape[0]
    # we do the scaling of ratio in the same way as in the SLIC paper
    # so the values have the same meaning
    ratio = (ratio / float(step)) ** 2
    print(ratio)
    cdef np.ndarray[dtype=np.float_t, ndim=3] image_yx = np.dstack([grid_y, grid_x, image / ratio]).copy("C")
    cdef int i, k, x, y, x_min, x_max, y_min, y_max, changes
    cdef double dist_mean

    cdef np.ndarray[dtype=np.int_t, ndim=2] nearest_mean = np.zeros((height, width), dtype=np.int)
    cdef np.ndarray[dtype=np.float_t, ndim=2] distance = np.ones((height, width), dtype=np.float) * np.inf
    cdef np.float_t* image_p = <np.float_t*> image_yx.data
    cdef np.float_t* distance_p = <np.float_t*> distance.data
    cdef np.float_t* current_distance
    cdef np.float_t* current_pixel
    cdef double tmp
    for i in xrange(max_iter):
        changes = 0
        print("iteration %d" % i)
        current_mean = <np.float_t*> means.data
        # assign pixels to means
        for k in xrange(n_means):
            # compute windows:
            y_min = int(max(current_mean[0] - 2 * step, 0))
            y_max = int(min(current_mean[0] + 2 * step, height))
            x_min = int(max(current_mean[1] - 2 * step, 0))
            x_max = int(min(current_mean[1] + 2 * step, height))
            for y in xrange(y_min, y_max):
                current_pixel = &image_p[5 * (y * width + x_min)]
                current_distance = &distance_p[y * width + x_min]
                for x in xrange(x_min, x_max):
                    mean_entry = current_mean
                    dist_mean = 0
                    for c in range(5):
                        # you would think the compiler can optimize this itself.
                        # mine can't (with O2)
                        tmp = current_pixel[0] - mean_entry[0]
                        dist_mean += tmp * tmp
                        current_pixel += 1
                        mean_entry += 1
                    # some precision issue here. Doesnt work if testing ">"
                    if current_distance[0] - dist_mean > 1e-10:
                        nearest_mean[y, x] = k
                        current_distance[0] = dist_mean
                        changes += 1
                    current_distance += 1
            current_mean += 5
            break
        # recompute means:
        means_list = [np.bincount(nearest_mean.ravel(), image_yx[:, :, j].ravel())
                for j in xrange(5)]
        in_mean = np.bincount(nearest_mean.ravel())
        means = (np.vstack(means_list) / in_mean).T.copy("C")
    return nearest_mean
