import numpy as np

from .util import pairwise_hamming_distance
from .match_cy import _binary_cross_check_loop


def match_binary_descriptors(keypoints1, descriptors1, keypoints2,
                             descriptors2, threshold=0.40, cross_check=True):
    """Match keypoints described using binary descriptors in one image to
    those in second image.

    Parameters
    ----------
    keypoints1 : (M, 2) ndarray
        M Keypoints from the first image described using skimage.feature.brief
    descriptors1 : (M, P) ndarray
        Binary descriptors of size P about M keypoints in the first image.
    keypoints2 : (N, 2) ndarray
        N Keypoints from the second image described using skimage.feature.brief
    descriptors2 : (N, P) ndarray
        Binary descriptors of size P about N keypoints in the second image.
    threshold : float in range [0, 1]
        Maximum allowable hamming distance between descriptors of two keypoints
        in separate images to be regarded as a match.
    cross_check : bool
        If True, the matched keypoints are returned after cross checking i.e. a
        matched pair (keypoint1, keypoint2) is returned iff keypoint2 is the best
        match for keypoint1 in second image and keypoint1 is the best match for
        keypoint2 in first image.

    Returns
    -------
    matches : (Q, 2, 2) ndarray
        Location of Q matched keypoint pairs from two images.
    mask1 : (Q,) ndarray
        Indices of keypoints in keypoints1 that have been matched.
    mask2 : (Q,) ndarray
        Indices of keypoints in keypoints2 that have been matched.

    """
    if (keypoints1.shape[0] != descriptors1.shape[0]
    or keypoints2.shape[0] != descriptors2.shape[0]):
        raise ValueError("The number of keypoints and number of described "
                         "keypoints do not match.")

    if descriptors1.shape[1] != descriptors2.shape[1]:
        raise ValueError("Descriptor sizes for matching keypoints in both "
                         "the images should be equal.")

    # Get hamming distances between keypoints1 and keypoints2
    distance = pairwise_hamming_distance(descriptors1, descriptors2)

    if cross_check:
        matched_keypoints1_index = np.argmin(distance, axis=1)
        matched_keypoints2_index = np.argmin(distance, axis=0)

        matched_index = _binary_cross_check_loop(matched_keypoints1_index,
                                                 matched_keypoints2_index,
                                                 distance, threshold)

        matches = np.zeros((matched_index.shape[0], 2, 2),
                                          dtype=np.intp)
        mask1 = matched_index[:, 0]
        mask2 = matched_index[:, 1]
        matches[:, 0, :] = keypoints1[mask1]
        matches[:, 1, :] = keypoints2[mask2]

    else:
        temp = distance > threshold
        row_check = np.any(~temp, axis=1)
        matched_keypoints2 = keypoints2[np.argmin(distance, axis=1)]
        matches = np.zeros((np.sum(row_check), 2, 2),
                                          dtype=np.intp)
        matches[:, 0, :] = keypoints1[row_check]
        matches[:, 1, :] = matched_keypoints2[row_check]
        mask1 = np.where(row_check == True)
        mask2 = np.argmin(distance, axis=1)[row_check]

    return matches, mask1, mask2
