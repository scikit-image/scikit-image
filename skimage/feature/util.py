import numpy as np
from scipy.spatial.distance import hamming


def _remove_border_keypoints(image, keypoints, dist):
    """Removes keypoints that are within dist pixels from the image border."""
    width = image.shape[0]
    height = image.shape[1]

    keypoints = keypoints[(dist - 1 < keypoints[:, 0])
                          & (keypoints[:, 0] < width - dist + 1)
                          & (dist - 1 < keypoints[:, 1])
                          & (keypoints[:, 1] < height - dist + 1)]

    return keypoints


def hamming_distance(array1, array2):
    """A dissimilarity measure used for matching keypoints in different images
    using binary feature descriptors like BRIEF etc.

    Parameters
    ----------
    array1 : (P1, D) array of dtype bool
        P1 vectors of size D with boolean elements.
    array2 : (P2, D) array of dtype bool
        P2 vectors of size D with boolean elements.

    Returns
    -------
    distance : (P1, P2) array of dtype float
        2D ndarray with value at an index (i, j) in the range [0, 1]
        representing the hamming distance between ith vector in
        array1 and jth vector in array2.

    """
    distance = np.zeros((array1.shape[0], array2.shape[0]), dtype=float)
    for i in range(array1.shape[0]):
        for j in range(array2.shape[0]):
            distance[i, j] = hamming(array1[i, :], array2[j, :])
    return distance
