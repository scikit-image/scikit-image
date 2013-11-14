import numpy as np
from scipy.ndimage.filters import gaussian_filter

from .util import (_mask_border_keypoints, pairwise_hamming_distance,
                   _prepare_grayscale_input_2D)

from ._brief_cy import _brief_loop


def descriptor_brief(image, keypoints, descriptor_size=256, mode='normal',
                     patch_size=49, sample_seed=1, variance=2):
    """Extract BRIEF Descriptor about given keypoints for a given image.

    Parameters
    ----------
    image : 2D ndarray
        Input image.
    keypoints : record array with P rows
        Record array with fields row, col, octave, orientation, response.
        Octave, orientation and response can be None.
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
    keypoints : record array with Q rows
        Record array with fields row, col, octave, orientation, response.
        Octave, orientation and response can be None.

    References
    ----------
    .. [1] Michael Calonder, Vincent Lepetit, Christoph Strecha and Pascal Fua
           "BRIEF : Binary robust independent elementary features",
           http://cvlabwww.epfl.ch/~lepetit/papers/calonder_eccv10.pdf

    Examples
    --------
    >>> from skimage.feature.corner import corner_peaks, corner_harris
    >>> from skimage.feature import (pairwise_hamming_distance, descriptor_brief,
    ...                              match_binary_descriptors,
    ...                              create_keypoint_recarray)
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
    >>> keypoints1 = create_keypoint_recarray(keypoints1[:, 0], keypoints1[:, 1])
    >>> descriptors1, keypoints1 = descriptor_brief(square1, keypoints1, patch_size=5)
    >>> keypoints1
    rec.array([(2.0, 2.0, nan, nan, nan), (2.0, 5.0, nan, nan, nan),
               (5.0, 2.0, nan, nan, nan), (5.0, 5.0, nan, nan, nan)], 
               dtype=[('row', '<f8'), ('col', '<f8'), ('octave', '<f8'),
               ('orientation', '<f8'), ('response', '<f8')])
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
    >>> keypoints2 = create_keypoint_recarray(keypoints2[:, 0], keypoints2[:, 1])
    >>> keypoints2
    rec.array([(2.0, 2.0, nan, nan, nan), (2.0, 6.0, nan, nan, nan),
               (6.0, 2.0, nan, nan, nan), (6.0, 6.0, nan, nan, nan)], 
               dtype=[('row', '<f8'), ('col', '<f8'), ('octave', '<f8'),
               ('orientation', '<f8'), ('response', '<f8')])
    >>> descriptors2, keypoints2 = descriptor_brief(square2, keypoints2, patch_size=5)
    >>> pairwise_hamming_distance(descriptors1, descriptors2)
    array([[ 0.03125  ,  0.3203125,  0.3671875,  0.6171875],
           [ 0.3203125,  0.03125  ,  0.640625 ,  0.375    ],
           [ 0.375    ,  0.6328125,  0.0390625,  0.328125 ],
           [ 0.625    ,  0.3671875,  0.34375  ,  0.0234375]])
    >>> matched_kpts, mask1, mask2 = match_binary_descriptors(keypoints1,
    ...                                                       descriptors1,
    ...                                                       keypoints2,
    ...                                                       descriptors2)
    >>> matched_kpts
    array([[[2, 2],
            [2, 2]],

           [[2, 5],
            [2, 6]],

           [[5, 2],
            [6, 2]],

           [[5, 5],
            [6, 6]]])

    """
    np.random.seed(sample_seed)

    image = _prepare_grayscale_input_2D(image)

    # Gaussian Low pass filtering to alleviate noise
    # sensitivity
    image = gaussian_filter(image, variance)

    image = np.ascontiguousarray(image)

    keypoints_loc = np.array(np.squeeze(np.dstack((keypoints.row, keypoints.col)))
                             + 0.5, dtype=np.intp, order='C')

    # Removing keypoints that are within (patch_size / 2) distance from the
    # image border
    border_mask = _mask_border_keypoints(image, keypoints_loc, patch_size // 2)
    keypoints = keypoints[border_mask]
    keypoints_loc = keypoints_loc[border_mask]
    keypoints_loc = np.ascontiguousarray(keypoints_loc)

    descriptors = np.zeros((keypoints_loc.shape[0], descriptor_size),
                           dtype=bool, order='C')

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
        samples = np.array(samples, dtype=np.int32)
        pos1, pos2 = np.split(samples, 2)

    pos1 = np.ascontiguousarray(pos1)
    pos2 = np.ascontiguousarray(pos2)

    _brief_loop(image, descriptors.view(np.uint8), keypoints_loc, pos1, pos2)

    return descriptors, keypoints
