import numpy as np
from scipy.ndimage.filters import gaussian_filter

from ..util import img_as_float
from .util import _remove_border_keypoints, pairwise_hamming_distance

from ._brief_cy import _brief_loop


def brief(image, keypoints, descriptor_size=256, mode='normal', patch_size=49,
          sample_seed=1, variance=2):
    """Extract BRIEF Descriptor about given keypoints for a given image.

    Parameters
    ----------
    image : 2D ndarray
        Input image.
    keypoints : (P, 2) ndarray
        Array of keypoint locations in the format (row, col).
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
        Seed for sampling the decision pixel-pairs. From a square window with
        length patch_size, pixel pairs are sampled using the `mode` parameter
        to build the descriptors using intensity comparison. The value of
        `sample_seed` should be the same for the images to be matched while
        building the descriptors. Default is 1.
    variance : float
        Variance of the Gaussian Low Pass filter applied on the image to
        alleviate noise sensitivity. Default is 2.

    Returns
    -------
    descriptors : (Q, `descriptor_size`) ndarray of dtype bool
        2D ndarray of binary descriptors of size `descriptor_size` about Q
        keypoints after filtering out border keypoints with value at an index
        (i, j) either being True or False representing the outcome
        of Intensity comparison about ith keypoint on jth decision pixel-pair.
    keypoints : (Q, 2) ndarray
        Location i.e. (row, col) of keypoints after removing out those that
        are near border.

    References
    ----------
    .. [1] Michael Calonder, Vincent Lepetit, Christoph Strecha and Pascal Fua
           "BRIEF : Binary robust independent elementary features",
           http://cvlabwww.epfl.ch/~lepetit/papers/calonder_eccv10.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.feature.corner import corner_peaks, corner_harris
    >>> from skimage.feature import pairwise_hamming_distance, brief, match_keypoints_brief
    >>> square1 = np.zeros([8, 8], dtype=np.int32)
    >>> square1[2:6, 2:6] = 1
    >>> square1
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
    >>> keypoints1 = corner_peaks(corner_harris(square1), min_distance=1)
    >>> keypoints1
    array([[2, 2],
           [2, 5],
           [5, 2],
           [5, 5]])
    >>> descriptors1, keypoints1 = brief(square1, keypoints1, patch_size=5)
    >>> keypoints1
    array([[2, 2],
           [2, 5],
           [5, 2],
           [5, 5]])
    >>> square2 = np.zeros([9, 9], dtype=np.int32)
    >>> square2[2:7, 2:7] = 1
    >>> square2
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
    >>> keypoints2 = corner_peaks(corner_harris(square2), min_distance=1)
    >>> keypoints2
    array([[2, 2],
           [2, 6],
           [6, 2],
           [6, 6]])
    >>> descriptors2, keypoints2 = brief(square2, keypoints2, patch_size=5)
    >>> keypoints2
    array([[2, 2],
           [2, 6],
           [6, 2],
           [6, 6]])
    >>> pairwise_hamming_distance(descriptors1, descriptors2)
    array([[ 0.03125  ,  0.3203125,  0.3671875,  0.6171875],
           [ 0.3203125,  0.03125  ,  0.640625 ,  0.375    ],
           [ 0.375    ,  0.6328125,  0.0390625,  0.328125 ],
           [ 0.625    ,  0.3671875,  0.34375  ,  0.0234375]])
    >>> match_keypoints_brief(keypoints1, descriptors1, keypoints2, descriptors2)
    array([[[ 2,  2],
            [ 2,  2]],

           [[ 2,  5],
            [ 2,  6]],

           [[ 5,  2],
            [ 6,  2]],

           [[ 5,  5],
            [ 6,  6]]])

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

    # Removing keypoints that are within (patch_size / 2) distance from the
    # image border
    keypoints = keypoints[_remove_border_keypoints(image, keypoints, patch_size // 2)]
    keypoints = np.ascontiguousarray(keypoints)

    descriptors = np.zeros((keypoints.shape[0], descriptor_size), dtype=bool,
                            order='C')

    # Sampling pairs of decision pixels in patch_size x patch_size window
    if mode == 'normal':

        samples = (patch_size / 5.0) * np.random.randn(descriptor_size * 8)
        samples = np.array(samples, dtype=np.int32)
        samples = samples[(samples < (patch_size // 2))
                          & (samples > - (patch_size - 2) // 2)]

        pos1 = samples[:descriptor_size * 2]
        pos1 = pos1.reshape(descriptor_size, 2)
        pos2 = samples[descriptor_size * 2:descriptor_size * 4]
        pos2 = pos2.reshape(descriptor_size, 2)

    else:

        samples = np.random.randint(-(patch_size - 2) // 2,
                                    (patch_size // 2) + 1,
                                    (descriptor_size * 2, 2))
        pos1, pos2 = np.split(samples, 2)

    pos1 = np.ascontiguousarray(pos1)
    pos2 = np.ascontiguousarray(pos2)

    _brief_loop(image, descriptors.view(np.uint8), keypoints, pos1, pos2)

    return descriptors, keypoints


def match_keypoints_brief(keypoints1, descriptors1, keypoints2,
                          descriptors2, threshold=0.15):
    """Match keypoints described using BRIEF descriptors in one image to
    those in second image.

    Parameters
    ----------
    keypoints1 : (M, 2) ndarray
        M Keypoints from the first image described using skimage.feature.brief
    descriptors1 : (M, P) ndarray
        BRIEF descriptors of size P about M keypoints in the first image.
    keypoints2 : (N, 2) ndarray
        N Keypoints from the second image described using skimage.feature.brief
    descriptors2 : (N, P) ndarray
        BRIEF descriptors of size P about N keypoints in the second image.
    threshold : float in range [0, 1]
        Maximum allowable hamming distance between descriptors of two keypoints
        in separate images to be regarded as a match. Default is 0.15.

    Returns
    -------
    match_keypoints_brief : (Q, 2, 2) ndarray
        Location of Q matched keypoint pairs from two images.

    """
    if (keypoints1.shape[0] != descriptors1.shape[0]
    or keypoints2.shape[0] != descriptors2.shape[0]):
        raise ValueError("The number of keypoints and number of described "
                         "keypoints do not match. Make the optional parameter "
                         "return_keypoints True to get described keypoints.")

    if descriptors1.shape[1] != descriptors2.shape[1]:
        raise ValueError("Descriptor sizes for matching keypoints in both "
                         "the images should be equal.")

    # Get hamming distances between keeypoints1 and keypoints2
    distance = pairwise_hamming_distance(descriptors1, descriptors2)

    temp = distance > threshold
    row_check = np.any(~temp, axis=1)
    matched_keypoints2 = keypoints2[np.argmin(distance, axis=1)]
    matched_keypoint_pairs = np.zeros((np.sum(row_check), 2, 2), dtype=np.intp)
    matched_keypoint_pairs[:, 0, :] = keypoints1[row_check]
    matched_keypoint_pairs[:, 1, :] = matched_keypoints2[row_check]

    return matched_keypoint_pairs
