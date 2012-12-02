import numpy as np
cimport numpy as np
from time import time
from scipy import ndimage
from ..util import img_as_float
from ..color import rgb2lab


def slic(image, n_segments=100, ratio=10., max_iter=10, sigma=1,
        convert2lab=True):
    """Segments image using k-means clustering in Color-(x,y) space.

    Parameters
    ----------
    image : (width, height, 3) ndarray
        Input image.
    n_segments : int
        The (approximate) number of labels in the segmented output image.
    ratio: float
        Balances color-space proximity and image-space proximity.
        Higher values give more weight to color-space.
    max_iter : int
        Maximum number of iterations of k-means.
    sigma : float
        Width of Gaussian smoothing kernel for preprocessing. Zero means no
        smoothing.
    convert2lab : bool
        Whether the input should be converted to Lab colorspace prior to
        segmentation.  For this purpose, the input is assumed to be RGB. Highly
        recommended.

    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.

    Notes
    -----
    The image is smoothed using a Gaussian kernel prior to segmentation.

    References
    ----------
    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.

    Examples
    --------
    >>> from skimage.segmentation import slic
    >>> from skimage.data import lena
    >>> img = lena()
    >>> segments = slic(img, n_segments=100, ratio=10)
    >>> # Increasing the ratio parameter yields more square regions
    >>> segments = slic(img, n_segments=100, ratio=20)
    """
    image = np.atleast_3d(image)
    if image.shape[2] != 3:
        ValueError("Only 3-channel 2D images are supported.")
    image = ndimage.gaussian_filter(img_as_float(image), [sigma, sigma, 0])
    if convert2lab:
        image = rgb2lab(image)

    # initialize on grid:
    cdef int height, width
    height, width = image.shape[:2]
    # approximate grid size for desired n_segments
    cdef int step = np.ceil(np.sqrt(height * width / n_segments))
    grid_y, grid_x = np.mgrid[:height, :width]
    means_y = grid_y[::step, ::step]
    means_x = grid_x[::step, ::step]

    means_color = np.zeros((means_y.shape[0], means_y.shape[1], 3))
    cdef np.ndarray[dtype=np.float_t, ndim=2] means \
            = np.dstack([means_y, means_x, means_color]).reshape(-1, 5)
    cdef np.float_t* current_mean
    cdef np.float_t* mean_entry
    n_means = means.shape[0]
    # we do the scaling of ratio in the same way as in the SLIC paper
    # so the values have the same meaning
    ratio = (ratio / float(step)) ** 2
    cdef np.ndarray[dtype=np.float_t, ndim=3] image_yx \
            = np.dstack([grid_y, grid_x, image / ratio]).copy("C")
    cdef int i, k, x, y, x_min, x_max, y_min, y_max, changes
    cdef double dist_mean

    cdef np.ndarray[dtype=np.int_t, ndim=2] nearest_mean \
            = np.zeros((height, width), dtype=np.int)
    cdef np.ndarray[dtype=np.float_t, ndim=2] distance \
            = np.empty((height, width))
    cdef np.float_t* image_p = <np.float_t*> image_yx.data
    cdef np.float_t* distance_p = <np.float_t*> distance.data
    cdef np.float_t* current_distance
    cdef np.float_t* current_pixel
    cdef double tmp
    for i in range(max_iter):
        distance.fill(np.inf)
        changes = 0
        current_mean = <np.float_t*> means.data
        # assign pixels to means
        for k in range(n_means):
            # compute windows:
            y_min = int(max(current_mean[0] - 2 * step, 0))
            y_max = int(min(current_mean[0] + 2 * step, height))
            x_min = int(max(current_mean[1] - 2 * step, 0))
            x_max = int(min(current_mean[1] + 2 * step, width))
            for y in range(y_min, y_max):
                current_pixel = &image_p[5 * (y * width + x_min)]
                current_distance = &distance_p[y * width + x_min]
                for x in range(x_min, x_max):
                    mean_entry = current_mean
                    dist_mean = 0
                    for c in range(5):
                        # you would think the compiler can optimize the squaring
                        # itself. mine can't (with O2)
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
        if changes == 0:
            break
        # recompute means:
        means_list = [np.bincount(nearest_mean.ravel(),
                      image_yx[:, :, j].ravel()) for j in range(5)]
        in_mean = np.bincount(nearest_mean.ravel())
        in_mean[in_mean == 0] = 1
        means = (np.vstack(means_list) / in_mean).T.copy("C")
    return nearest_mean
