import scipy.sparse as sparse
import numpy as np

__all__ = ['contingency_table']


def contingency_table(im_true, im_test, *, ignore_labels=(), normalize=False):
    """
    Return the contingency table for all regions in matched segmentations.

    Parameters
    ----------
    im_true : ndarray of int
        Ground-truth label image, same shape as im_test.
    im_test : ndarray of int
        Test image.
    ignore_labels : sequence of int, optional
        Labels to ignore. Any part of the true image labeled with any of these
        values will not be counted in the score.
    normalize : bool
        Determines if the contingency table is normalized by pixel count.

    Returns
    -------
    cont : scipy.sparse.csr_matrix
        A contingency table. `cont[i, j]` will equal the number of voxels
        labeled `i` in `im_true` and `j` in `im_test`.
    """

    im_test_r = im_test.ravel()
    im_true_r = im_true.ravel()
    ignored = np.zeros(im_true_r.shape, np.bool)
    for label in ignore_labels:
        ignored[im_true_r == label] = True
    data = np.ones(im_true_r.shape)
    data[ignored] = 0
    if normalize:
        data = data / im_true.size
    cont = sparse.coo_matrix((data, (im_true_r, im_test_r))).tocsr()
    return cont
