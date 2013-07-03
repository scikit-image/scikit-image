import numpy as np
from scipy.spatial.distance import hamming


def _remove_border_keypoints(image, keypoints, dist):
	"""Removes keypoints that are within dist pixels from the image border.
	"""
    width = image.shape[0]
    height = image.shape[1]

    keypoints = keypoints[(dist < keypoints[:, 0])
                          & (keypoints[:, 0] < width - dist)
                          & (dist < keypoints[:, 1])
                          & (keypoints[:, 1] < height - dist)]

    return keypoints


def hamming_distance(descriptors1, descriptors2):
    """A dissimilarity measure used for matching keypoints in different images
    using binary feature descriptors like BRIEF etc.

    Parameters
    ----------
    descriptors1 : (P1, D) array of dtype bool
        Binary feature descriptors for keypoints in the first image.
        2D ndarray with a binary descriptors of size D about P1 keypoints
        with value at an index (i, j) either being True or False representing
        the outcome of Intensity comparison about ith keypoint on jth decision
        pixel-pair.
    descriptors2 : (P2, D) array of dtype bool
        Binary feature descriptors for keypoints in the second image.
        2D ndarray with a binary descriptors of size D about P2 keypoints
        with value at an index (i, j) either being True or False representing
        the outcome of Intensity comparison about ith keypoint on jth decision
        pixel-pair.

    Returns
    -------
    distance : (P1, P2) array of dtype float
        2D ndarray with value at an index (i, j) in the range [0, 1]
        representing the extent of dissimilarity between ith keypoint of in
        first image and jth keypoint in second image.

    """
    distance = np.zeros((descriptors1.shape[0], descriptors2.shape[0]), dtype=float)
    for i in range(descriptors1.shape[0]):
        for j in range(descriptors2.shape[0]):
            distance[i, j] = hamming(descriptors1[i, :], descriptors2[j, :])
    return distance
