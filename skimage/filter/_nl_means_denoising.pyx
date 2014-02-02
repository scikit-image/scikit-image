import numpy as np
from skimage import util
cimport numpy as np
cimport cython
from libc.math cimport exp

DTYPE = np.float
ctypedef np.float32_t DTYPE_t


@cython.boundscheck(False)
cdef inline float patch_distance2d(DTYPE_t [:, :] p1,
                                   DTYPE_t [:, :] p2,
                                   DTYPE_t [:, ::] w, int s):
    cdef int i, j
    cdef float distance = 0
    cdef float tmp_diff
    for i in range(s):
        for j in range(s):
            tmp_diff = p1[i, j] - p2[i, j]
            distance += w[i, j] * tmp_diff * tmp_diff
    distance = exp(- distance)
    return distance


@cython.boundscheck(False)
cdef inline float patch_distance(DTYPE_t [:, :, :] p1,
                                 DTYPE_t [:, :, :] p2,
                                 DTYPE_t [:, :, ::] w, int s):
    cdef int i, j, k
    cdef float distance = 0
    cdef float tmp_diff
    for i in range(s):
        for j in range(s):
            for k in range(s):
                tmp_diff = p1[i, j, k] - p2[i, j, k]
                distance += w[i, j, k] * tmp_diff * tmp_diff
    distance = exp(- distance)
    return distance


@cython.cdivision(True)
@cython.boundscheck(False)
def _nl_means_denoising_2d(image, int s=7, int d=13, float h=0.1):
    """
    Perform non-local means denoising on 2-D array

    Parameters
    ----------
    image: ndarray
        input data to be denoised

    s: int, optional
        size of patches used for denoising

    d: int, optional
        maximal distance in pixels where to search patches used for denoising

    h: float, optional
        cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.
    """
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef int n_x, n_y
    n_x, n_y = image.shape
    cdef int offset = s / 2
    cdef DTYPE_t [:, ::1] padded = np.ascontiguousarray(util.pad(image,
                                offset, mode='reflect').astype(np.float32))
    cdef DTYPE_t [:, ::1] result = padded.copy()
    h *= (np.max(padded) - np.min(padded))
    cdef float A = ((s - 1.) / 4.)
    cdef float new_value
    cdef float weight_sum, weight
    xg, yg = np.mgrid[-offset:offset + 1, -offset:offset + 1]
    cdef DTYPE_t [:, ::1] w = np.ascontiguousarray(np.exp(
                                    - (xg ** 2 + yg ** 2)/(2 * A ** 2)).
                                    astype(np.float32))
    cdef float distance
    cdef int x, y, i, j
    w = 1./ (np.sum(w) * 2 * h ** 2) * w
    for x in range(offset, n_x + offset):
        for y in range(offset, n_y + offset):
            new_value = 0
            weight_sum = 0
            for i in range(max(- d, offset - x),
                           min(d + 1, n_x - x - 1)):
                for j in range(max(- d, offset - y),
                               min(d + 1, n_y - y - 1)):
                    weight = patch_distance2d(
                                padded[x - offset: x + offset + 1,
                                        y - offset: y + offset + 1],
                                padded[x + i - offset: x + i + offset + 1,
                                        y + j - offset: y + j + offset + 1],
                                        w, s)
                    weight_sum += weight
                    new_value += weight * padded[x + i, y + j]
            result[x, y] = new_value / weight_sum
    return result[offset:-offset, offset:-offset]


@cython.cdivision(True)
@cython.boundscheck(False)
def _nl_means_denoising_3d(image, int s=7,
            int d=13, float h=0.1):
    """
    Perform non-local means denoising on 3-D array

    Parameters
    ----------
    image: ndarray
        input data to be denoised

    s: int, optional
        size of patches used for denoising

    d: int, optional
        maximal distance in pixels where to search patches used for denoising

    h: float, optional
        cut-off distance (in gray levels)
    """
    if s % 2 == 0:
        s += 1  # odd value for symmetric patch
    cdef int n_x, n_y, n_z
    n_x, n_y, n_z = image.shape
    cdef int offset = s / 2
    cdef DTYPE_t [:, :, ::1] padded = np.ascontiguousarray(util.pad(image,
                                offset, mode='reflect').astype(np.float32))
    cdef DTYPE_t [:, :, ::1] result = padded.copy()
    h *= (np.max(padded) - np.min(padded))
    cdef float A = ((s - 1.) / 4.)
    cdef float new_value
    cdef float weight_sum, weight
    xg, yg, zg = np.mgrid[-offset: offset + 1, -offset: offset+1,
                            -offset: offset + 1]
    cdef DTYPE_t [:, :, ::1] w = np.ascontiguousarray(np.exp(
                                - (xg ** 2 + yg ** 2 + zg ** 2)/(2 * A ** 2)).
                                astype(np.float32))
    cdef float distance
    cdef int x, y, z, i, j, k
    w = 1./ (np.sum(w) * 2 * h ** 2) * w
    for x in range(offset, n_x + offset):
        for y in range(offset, n_y + offset):
            for z in range(offset, n_z + offset):
                new_value = 0
                weight_sum = 0
                for i in range(max(- d, offset - x),
                              min(d + 1, n_x - x - 1)):
                    for j in range(max(- d, offset - y),
                                   min(d + 1, n_y - y - 1)):
                        for k in range(max(- d, offset - z),
                                   min(d + 1, n_z - z - 1)):
                            weight = patch_distance(
                                    padded[x - offset: x + offset +1,
                                                y - offset: y + offset +1,
                                                z - offset: z + offset +1],
                                    padded[x + i - offset: x + i + offset +1,
                                            y + j - offset: y + j + offset +1,
                                            z + k - offset: z + k + offset +1],
                                    w, s)
                            weight_sum += weight
                            new_value += weight * padded[x + i, y + j, z + k]
                result[x, y, z] = new_value / weight_sum
    return result[offset:-offset, offset:-offset, offset:-offset]


def nl_means_denoising(image, patch_size=7, patch_distance=11, h=0.1):
    """
    Perform non-local means denoising on 2-D or 3-D grayscale arrays

    Parameters
    ----------
    image: ndarray
        input data to be denoised

    patch_size: int, optional
        size of patches used for denoising

    patch_distance: int, optional
        maximal distance in pixels where to search patches used for denoising

    h: float, optional
        cut-off distance (in gray levels). The higher h, the more permissive
        one is in accepting patches.

    Returns
    -------

    result: ndarray
        denoised image, of same shape as `image`.

    Notes
    -----

    The non-local means algorithm is well suited for denoising images with
    specific textures. The principle of the algorithm is to average the value
    of a given pixel with values of other pixels in a limited neighbourhood,
    provided that the *patches* centered on the other pixels are similar enough
    to the patch centered on the pixel of interest. 

    The complexity of the algorithm is

    image.size * patch_size ** image.ndim * patch_distance ** image.ndim

    Hence, changing the size of patches or their maximal distance has a
    strong effect on computing times, especially for 3-D images.

    The image is padded using the `reflect` mode of `skimage.util.pad`
    before denoising.

    References
    ----------
    .. [1] Buades, A., Coll, B., & Morel, J. M. (2005, June). A non-local
        algorithm for image denoising. In CVPR 2005, Vol. 2, pp. 60-65, IEEE.

    Examples
    --------
    >>> a = np.zeros((40, 40))
    >>> a[10:-10, 10:-10] = 1.
    >>> a += 0.3*np.random.randn(*a.shape)
    >>> denoised_a = nl_means_denoising(a, 7, 5, 0.1)
    """
    if image.ndim == 2:
        return np.array(_nl_means_denoising_2d(image, patch_size,
                                patch_distance, h))
    if image.ndim == 3 and image.shape[-1] > 4:  # only grayscale
        return np.array(_nl_means_denoising_3d(image, patch_size,
                                patch_distance, h))
    else:
        raise ValueError("Non local means denoising is only possible for \
        2D and 3-D grayscale images.")
