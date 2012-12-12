import numpy as np

def join_segmentations(s1, s2):
    """Return the join of the two input segmentations.

    The join J of S1 and S2 is defined as the segmentation in which two voxels
    are in the same segment in J if and only if they are in the same segment
    in *both* S1 and S2.

    Parameters
    ----------
    s1, s2 : numpy arrays
        s1 and s2 are label fields of the same shape.

    Returns
    -------
    j : numpy array
        The join segmentation of s1 and s2.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.segmentation import join_segmentations
    >>> s1 = np.array([[0, 0, 1, 1],
    ...                [0, 2, 1, 1],
    ...                [2, 2, 2, 1]])
    >>> s2 = np.array([[0, 1, 1, 0],
    ...                [0, 1, 1, 0],
    ...                [0, 1, 1, 1]])
    >>> join_segmentations(s1, s2)
    array([[0, 1, 3, 2],
           [0, 5, 3, 2],
           [4, 5, 5, 3]])
    """
    if s1.shape != s2.shape:
        raise ValueError("Cannot join segmentations of different shape. " +
                         "s1.shape: %s, s2.shape: %s" % (s1.shape, s2.shape))
    s1 = relabel_from_one(s1)[0]
    s2 = relabel_from_one(s2)[0]
    j = (s2.max() + 1) * s1 + s2
    j = relabel_from_one(j)[0]
    return j

def relabel_from_one(ar):
    """Convert array ar of arbitrary labels to labels 1...len(np.unique(ar))+1

    This function also returns the forward map (mapping the original labels to
    the reduced labels) and the inverse map (mapping the reduced labels back
    to the original ones).

    Parameters
    ----------
    ar : numpy ndarray (integer type)

    Returns
    -------
    relabeled : numpy array of same shape as ar
    forward_map : 1d numpy array of length np.unique(ar) + 1
    inverse_map : 1d numpy array of length len(np.unique(ar))
        The length is len(np.unique(ar)) + 1 if 0 is not in np.unique(ar)

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.segmentation import relabel_from_one
    >>> ar = array([1, 1, 5, 5, 8, 99, 42])
    >>> relab, fw, inv = relabel_from_one(ar)
    >>> relab
    array([1, 1, 2, 2, 3, 5, 4])
    >>> fw
    array([0, 1, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 5])
    >>> inv
    array([ 0,  1,  5,  8, 42, 99])
    >>> (fw[ar] == relab).all()
    True
    >>> (inv[relab] == ar).all()
    True
    """
    labels = np.unique(ar)
    labels0 = labels[labels != 0]
    m = labels.max()
    if m == len(labels0): # nothing to do, already 1...n labels
        return ar, labels, labels
    forward_map = np.zeros(m+1, int)
    forward_map[labels0] = np.arange(1, len(labels0) + 1)
    if not (labels == 0).any():
        labels = np.concatenate(([0], labels))
    inverse_map = labels
    return forward_map[ar], forward_map, inverse_map
