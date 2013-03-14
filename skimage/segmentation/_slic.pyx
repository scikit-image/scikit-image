#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import collections as coll
import numpy as np
from time import time
from scipy import ndimage

cimport numpy as cnp

from ..util import img_as_float, regular_grid
from ..color import rgb2lab, gray2rgb


def slic(image, n_segments=100, ratio=10., max_iter=10, sigma=1,
         multichannel=True, convert2lab=True):
    """Segments image using k-means clustering in Color-(x,y) space.

    Parameters
    ----------
    image : (width, height [, depth] [, 3]) ndarray
        Input image, which can be 2D or 3D, and grayscale or multi-channel
        (see `multichannel` parameter).
    n_segments : int, optional (default: 100)
        The (approximate) number of labels in the segmented output image.
    ratio: float, optional (default: 10)
        Balances color-space proximity and image-space proximity.
        Higher values give more weight to color-space.
    max_iter : int, optional (default: 10)
        Maximum number of iterations of k-means.
    sigma : float, optional (default: 1)
        Width of Gaussian smoothing kernel for preprocessing. Zero means no
        smoothing.
    multichannel : bool, optional (default: True)
        Whether the last axis of the image is to be interpreted as multiple
        channels. Only 3 channels are supported.
    convert2lab : bool, optional (default: True)
        Whether the input should be converted to Lab colorspace prior to
        segmentation.  For this purpose, the input is assumed to be RGB. Highly
        recommended.

    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.

    Raises
    ------
    ValueError
        If:
            - the image dimension is not 2 or 3 and `multichannel == False`, OR
            - the image dimension is not 3 or 4 and `multichannel == True`, OR
            - `multichannel == True` and the length of the last dimension of
            the image is not 3.

    Notes
    -----
    The image is optionally smoothed using a Gaussian kernel prior to
    segmentation.

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
    if ((not multichannel and image.ndim not in [2, 3]) or
            (multichannel and image.ndim not in [3, 4]) or
            (multichannel and image.shape[-1] != 3)):
        ValueError("Only 1- or 3-channel 2- or 3-D images are supported.")
    if image.ndim in [2, 3] and not multichannel:
        image = gray2rgb(image)
    if image.ndim == 3:
        # See 2D RGB image as 3D RGB image with Z = 1
        image = image[np.newaxis, ...]
    if not isinstance(sigma, coll.Iterable):
        sigma = np.array([sigma, sigma, sigma, 0])
    if (sigma > 0).any():
        image = ndimage.gaussian_filter(img_as_float(image), sigma)
    if convert2lab:
        image = rgb2lab(image)

    # initialize on grid:
    cdef Py_ssize_t depth, height, width
    depth, height, width = image.shape[:3]
    # approximate grid size for desired n_segments
    cdef Py_ssize_t step_z, step_y, step_x
    grid_z, grid_y, grid_x = np.mgrid[:depth, :height, :width]
    slices = regular_grid(image.shape, n_segments)
    step_z, step_y, step_x = [int(s.step) for s in slices]
    means_z = grid_z[slices]
    means_y = grid_y[slices]
    means_x = grid_x[slices]

    means_color = np.zeros(means_z.shape + (3,))
    cdef cnp.ndarray[dtype=cnp.float_t, ndim=2] means = \
            np.concatenate([
                            means_z[..., np.newaxis],
                            means_y[..., np.newaxis],
                            means_x[..., np.newaxis],
                            means_color
                           ], axis=-1).reshape(-1, 6)
    cdef cnp.float_t* current_mean
    cdef cnp.float_t* mean_entry
    n_means = means.shape[0]
    # we do the scaling of ratio in the same way as in the SLIC paper
    # so the values have the same meaning
    ratio = (ratio / float(max((step_z, step_y, step_x)))) ** 2
    cdef cnp.ndarray[dtype=cnp.float_t, ndim=4] image_zyx \
            = np.concatenate([
                              grid_y[..., np.newaxis],
                              grid_x[..., np.newaxis],
                              grid_z[..., np.newaxis],
                              image / ratio
                             ], axis=-1).copy("C")
    cdef Py_ssize_t i, k, x, y, z, x_min, x_max, y_min, y_max, z_min, z_max, \
            changes
    cdef double dist_mean

    cdef cnp.ndarray[dtype=cnp.intp_t, ndim=3] nearest_mean \
            = np.zeros((depth, height, width), dtype=np.intp)
    cdef cnp.ndarray[dtype=cnp.float_t, ndim=3] distance \
            = np.empty((depth, height, width))
    cdef cnp.float_t* image_p = <cnp.float_t*> image_zyx.data
    cdef cnp.float_t* distance_p = <cnp.float_t*> distance.data
    cdef cnp.float_t* current_distance
    cdef cnp.float_t* current_pixel
    cdef double tmp
    for i in range(max_iter):
        distance.fill(np.inf)
        changes = 0
        current_mean = <cnp.float_t*> means.data
        # assign pixels to means
        for k in range(n_means):
            # compute windows:
            z_min = int(max(current_mean[0] - 2 * step_z, 0))
            z_max = int(min(current_mean[0] + 2 * step_z, depth))
            y_min = int(max(current_mean[1] - 2 * step_y, 0))
            y_max = int(min(current_mean[1] + 2 * step_y, height))
            x_min = int(max(current_mean[2] - 2 * step_x, 0))
            x_max = int(min(current_mean[2] + 2 * step_x, width))
            for z in range(z_min, z_max):
                for y in range(y_min, y_max):
                    current_pixel = \
                            &image_p[5 * ((z * height + y) * width + x_min)]
                    current_distance = \
                            &distance_p[(z * height + y) * width + x_min]
                    for x in range(x_min, x_max):
                        mean_entry = current_mean
                        dist_mean = 0
                        for c in range(5):
                            # you would think the compiler can optimize the
                            # squaring itself. mine can't (with O2)
                            tmp = current_pixel[0] - mean_entry[0]
                            dist_mean += tmp * tmp
                            current_pixel += 1
                            mean_entry += 1
                        # some precision issue here. Doesnt work if testing ">"
                        if current_distance[0] - dist_mean > 1e-10:
                            nearest_mean[z, y, x] = k
                            current_distance[0] = dist_mean
                            changes += 1
                        current_distance += 1
            current_mean += 6
        if changes == 0:
            break
        # recompute means:
        means_list = [np.bincount(nearest_mean.ravel(),
                      image_zyx[:, :, :, j].ravel()) for j in range(6)]
        in_mean = np.bincount(nearest_mean.ravel())
        in_mean[in_mean == 0] = 1
        means = (np.vstack(means_list) / in_mean).T.copy("C")
    return nearest_mean
