from .simple_metrics import _assert_compatible
import numpy as np
from ._contingency_table import contingency_table

__all__ = ['adapted_rand_error']

def adapted_rand_error(im_true=None, im_test=None,*,table=None, ignore_labels=None, normalize=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest. [1]_
    Parameters
    ----------
    im_true : ndarray of int
        Ground-truth label image.
    im_test : ndarray of int
        Test image.
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float
        The adapted Rand precision: this is the number of pairs of pixels that
        have the same label in the test label image *and* in the true image,
        divided by the number in the test image.
    rec : float
        The adapted Rand recall: this is the number of pairs of pixels that
        have the same label in the test label image *and* in the true image,
        divided by the number in the true image.
    Notes
    -----
    Pixels with label 0 in the true segmentation are ignored in the score.
    References
    ----------
    .. [1] Arganda-Carreras I, Turaga SC, Berger DR, et al. (2015)
           Crowdsourcing the creation of image segmentation algorithms
           for connectomics. Front. Neuroanat. 9:142.
           :DOI:`10.3389/fnana.2015.00142`
    """
    _assert_compatible(im_true, im_test)

    if table is None:
        p_ij = contingency_table(im_true, im_test, ignore_labels=[0], normalize=normalize)
    else:
        p_ij = table

    # Sum of the joint distribution squared
    sum_p_ij2 = p_ij.data @ p_ij.data - p_ij.sum()

    a_i = p_ij.sum(axis=1).A.ravel()
    b_i = p_ij.sum(axis=0).A.ravel()

    # Sum of squares of the test segment sizes (this is 2x the number of pairs
    # of pixels with the same label in im_test)
    sum_a2 = a_i @ a_i - a_i.sum()
    # Same for im_true
    sum_b2 = b_i @ b_i - b_i.sum()

    precision = sum_p_ij2 / sum_a2
    recall = sum_p_ij2 / sum_b2

    fscore = 2. * precision * recall / (precision + recall)
    are = 1. - fscore

    return are, precision, recall