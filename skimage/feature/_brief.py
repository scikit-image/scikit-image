import numpy as np
from scipy.ndimage.filters import gaussian_filter

from ..util import img_as_float
from .util import _remove_border_keypoints

from ._brief_cy import _brief_loop


def brief(image, keypoints, descriptor_size=256, mode='normal', patch_size=49,
          sample_seed=1, variance=2, return_keypoints=False):
    """Extract BRIEF Descriptor about given keypoints for a given image.

    Parameters
    ----------
    image : 2D ndarray
        Input image.
    keypoints : (P, 2) ndarray
        Array of keypoint locations.
    descriptor_size : int
        Size of BRIEF descriptor about each keypoint. Sizes 128, 256 and 512
        preferred by the authors. Default is 256.
    mode : string
        Probability distribution for sampling location of decision pixel-pairs
        around keypoints. Default is 'normal' otherwise uniform.
    patch_size : int
        Length of the two dimensional square patch sampling region around
        the keypoints. Default is 49.
    sample_seed : int
        Seed for sampling the decision pixel-pairs. Default is 1.
    return_keypoints : bool
        If True, return the Q keypoints (after filtering out the border
        keypoints) about which the descriptors are extracted. Default is False.

    Returns
    -------
    descriptors : (Q, descriptor_size) ndarray of dtype bool
        2D ndarray of binary descriptors of size descriptor_size about Q
        keypoints after filtering out border keypoints with value at an index
        (i, j) either being True or False representing the outcome
        of Intensity comparison about ith keypoint on jth decision pixel-pair.
    keypoints : (Q, 2) ndarray
        Keypoints after removing out those that are near border.
        Returned only if return_keypoints is True.

    References
    ----------
    .. [1] Michael Calonder, Vincent Lepetit, Christoph Strecha and Pascal Fua
    "BRIEF : Binary robust independent elementary features",
    http://cvlabwww.epfl.ch/~lepetit/papers/calonder_eccv10.pdf

    Examples
    --------
    >>> from skimage.feature.corner import *
    >>> from skimage.feature import brief, hamming_distance
    >>> square1 = np.zeros([10, 10])
    >>> square1[2:8, 2:8] = 1
    >>> square1
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    >>> keypoints1 = corner_peaks(corner_harris(square1), min_distance=1)
    >>> keypoints1
    array([[2, 2],
           [2, 7],
           [7, 2],
           [7, 7]])
    >>> descriptors1 = brief(square1, keypoints1, patch_size = 5)
    >>> square2 = np.zeros([12, 12])
    >>> square2[3:9, 3:9] = 1
    >>> square2
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    >>> keypoints2 = corner_peaks(corner_harris(square2), min_distance=1)
    >>> keypoints2
    array([[3, 3],
           [3, 8],
           [8, 3],
           [8, 8]])
    >>> descriptors2 = brief(square2, keypoints1, patch_size = 5)
    >>> hamming_distance(descriptors1, descriptors2)
    array([[ 0.02734375,  0.2890625 ,  0.32421875,  0.6171875 ],
           [ 0.3359375 ,  0.05078125,  0.6640625 ,  0.37109375],
           [ 0.359375  ,  0.64453125,  0.03125   ,  0.33203125],
           [ 0.640625  ,  0.40234375,  0.3828125 ,  0.01953125]])

    """

    np.random.seed(sample_seed)

    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")

    image = img_as_float(image)

    # Gaussian Low pass filtering to alleviate noise
    # sensitivity
    image = gaussian_filter(image, variance)

    image = np.ascontiguousarray(image)

    keypoints = np.array(keypoints + 0.5, dtype=np.intp, order='C')

    # Removing keypoints that are (patch_size / 2) distance from the image
    # border
    keypoints = _remove_border_keypoints(image, keypoints, patch_size / 2)

    descriptors = np.zeros((keypoints.shape[0], descriptor_size),
                           dtype=bool, order='C')

    # Sampling pairs of decision pixels in patch_size x patch_size window
    if mode == 'normal':

        samples = (patch_size / 5) * np.random.randn(descriptor_size * 8)
        samples = np.array(samples, dtype=np.int32)
        samples = samples[(samples < (patch_size / 2))
                          & (samples > - (patch_size - 1) / 2)]

        pos1 = samples[:descriptor_size * 2]
        pos1 = pos1.reshape(descriptor_size, 2)
        pos2 = samples[descriptor_size * 2:descriptor_size * 4]
        pos2 = pos2.reshape(descriptor_size, 2)

    else:

        samples = np.random.randint(-patch_size / 2, (patch_size / 2) + 1,
                                    (descriptor_size * 2, 2))
        pos1, pos2 = np.split(samples, 2)

    pos1 = np.ascontiguousarray(pos1)
    pos2 = np.ascontiguousarray(pos2)

    _brief_loop(image, descriptors.view(np.uint8), keypoints, pos1, pos2)

    if return_keypoints:
        return descriptors, keypoints
    else:
        return descriptors
