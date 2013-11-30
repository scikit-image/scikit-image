import numpy as np
from scipy.spatial.distance import cdist


def match_descriptors(descriptors1, descriptors2, metric=None, p=2,
                      threshold=0, cross_check=True):
    """Brute-force matching of descriptors.

    For each descriptor in the first set this matcher finds the closest
    descriptor in the second set (and vice-versa in the case of enabled
    cross-checking).

    Parameters
    ----------
    descriptors1 : (M, P) array
        Binary descriptors of size P about M keypoints in the first image.
    descriptors2 : (N, P) array
        Binary descriptors of size P about N keypoints in the second image.
    metric : {'euclidean', 'cityblock', 'minkowski', 'hamming', ...}
        The metric to compute the distance between two descriptors. See
        `scipy.spatial.distance.cdist` for all possible types. The hamming
        distance should be used for binary descriptors. By default the L2-norm
        is used for all descriptors of dtype float or double and the Hamming
        distance is used for binary descriptors automatically.
    p : int
        The p-norm to apply for ``metric='minkowski'``.
    threshold : float
        Maximum allowed distance between descriptors of two keypoints
        in separate images to be regarded as a match.
    cross_check : bool
        If True, the matched keypoints are returned after cross checking i.e. a
        matched pair (keypoint1, keypoint2) is returned if keypoint2 is the
        best match for keypoint1 in second image and keypoint1 is the best
        match for keypoint2 in first image.

    Returns
    -------
    indices1 : (Q, ) array
        Indices of corresponding matches for first set of descriptors.
    indices2 : (Q, ) array
        Indices of corresponding matches for second set of descriptors.

    """

    if descriptors1.shape[1] != descriptors2.shape[1]:
        raise ValueError("Descriptor length must equal.")

    if metric is None:
        if np.issubdtype(descriptors1.dtype, np.bool):
            metric = 'hamming'
        else:
            metric = 'euclidean'

    distances = cdist(descriptors1, descriptors2, metric=metric, p=p)

    indices1 = np.arange(descriptors1.shape[0])
    indices2 = np.argmin(distances, axis=1)

    if cross_check:
        matches1 = np.argmin(distances, axis=0)
        mask = indices1 == matches1[indices2]
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    return indices1, indices2
