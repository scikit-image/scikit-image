
import numpy as np
from ..util.dtype import dtype_range
from .._shared.utils import skimage_deprecation, warn

__all__ = ['compare_mse',
           'compare_nrmse',
           'compare_psnr',
           ]


def _assert_compatible(im1, im2):
    """Raise an error if the shape and dtype do not match."""
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return


def _as_floats(im1, im2):
    """Promote im1, im2 to nearest appropriate floating point precision."""
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2


def compare_mse(im1, im2):
    """Compute the mean-squared error between two images.

    Parameters
    ----------
    im1, im2 : ndarray
        Image.  Any dimensionality.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    """
    _assert_compatible(im1, im2)
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def compare_nrmse(im_true, im_test, norm_type='Euclidean'):
    """Compute the normalized root mean-squared error (NRMSE) between two
    images.

    Parameters
    ----------
    im_true : ndarray
        Ground-truth image.
    im_test : ndarray
        Test image.
    norm_type : {'Euclidean', 'min-max', 'mean'}
        Controls the normalization method to use in the denominator of the
        NRMSE.  There is no standard method of normalization across the
        literature [1]_.  The methods available here are as follows:

        - 'Euclidean' : normalize by the averaged Euclidean norm of
          ``im_true``::

              NRMSE = RMSE * sqrt(N) / || im_true ||

          where || . || denotes the Frobenius norm and ``N = im_true.size``.
          This result is equivalent to::

              NRMSE = || im_true - im_test || / || im_true ||.

        - 'min-max'   : normalize by the intensity range of ``im_true``.
        - 'mean'      : normalize by the mean of ``im_true``

    Returns
    -------
    nrmse : float
        The NRMSE metric.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Root-mean-square_deviation

    """
    _assert_compatible(im_true, im_test)
    im_true, im_test = _as_floats(im_true, im_test)

    norm_type = norm_type.lower()
    if norm_type == 'euclidean':
        denom = np.sqrt(np.mean((im_true*im_true), dtype=np.float64))
    elif norm_type == 'min-max':
        denom = im_true.max() - im_true.min()
    elif norm_type == 'mean':
        denom = im_true.mean()
    else:
        raise ValueError("Unsupported norm_type")
    return np.sqrt(compare_mse(im_true, im_test)) / denom


def compare_psnr(im_true, im_test, data_range=None):
    """ Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    im_true : ndarray
        Ground-truth image.
    im_test : ndarray
        Test image.
    data_range : int
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.

    Returns
    -------
    psnr : float
        The PSNR metric.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    """
    _assert_compatible(im_true, im_test)

    if data_range is None:
        if im_true.dtype != im_test.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "im_true.")
        dmin, dmax = dtype_range[im_true.dtype.type]
        true_min, true_max = np.min(im_true), np.max(im_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "im_true has intensity values outside the range expected for "
                "its data type.  Please manually specify the data_range")
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    im_true, im_test = _as_floats(im_true, im_test)

    err = compare_mse(im_true, im_test)
    return 10 * np.log10((data_range ** 2) / err)


def confusion_matrix(data_reference, data_test):
    """Compares reference and test data to generate a confusion matrix.

    Parameters
    ----------
    data_reference : ndarray
        Reference binary data (ground truth).
    data_test : ndarray
        Test binary data.

    Returns
    -------
    conf_matrix : array
        Matrix containing the number of true positives, false positives,
    false negatives, and true negatives.

    Notes
    -----
    The values true positive, false positive, false negative, and false
    positive are events obtained in the comparison between data_reference
    and data_test:

                   data_reference:        True                False
    data_test:
                            True       True positive   |   False positive
                                       ----------------------------------
                            False      False negative  |    True negative

    References
    ----------
    .. [1] Fawcett T. (2006) "An Introduction to ROC Analysis." Pattern
    Recognition Letters, 27 (8): 861-874, :DOI:`10.1016/j.patrec.2005.10.010`

    .. [2] Google Developers. "Machine Learning Crash Course with TensorFlow
    APIs: Classification: True vs. False and Positive vs. Negative." Available
    at: https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative

    """
    _assert_compatible(data_reference, data_test)

    true_pos = (data_reference & data_test).sum() / data_reference.size
    false_pos = (~data_reference & data_test).sum() / data_reference.size
    false_neg = (data_reference & ~data_test).sum() / data_reference.size
    true_neg = (~data_reference & ~data_test).sum() / data_reference.size

    return np.array([[true_pos, false_pos], [false_neg, true_neg]])


def measure_accuracy(conf_matrix):
    """
    """
    tr_pos, fl_pos, fl_neg, tr_neg = conf_matrix.ravel()
    return (tr_pos + tr_neg) / (tr_pos + tr_neg + fl_pos + fl_neg)


def measure_dice(conf_matrix):
    """
    """
    tr_pos, fl_pos, fl_neg, tr_neg = conf_matrix.ravel()
    return 2*tr_pos / (2*tr_pos + fl_pos + fl_neg)


def measure_informedness(conf_matrix):
    """
    """
    return measure_recall(conf_matrix) + measure_specificity(conf_matrix) - 1


def measure_matthews(conf_matrix):
    """
    """
    tr_pos, fl_pos, fl_neg, tr_neg = conf_matrix.ravel()
    mcc = (tr_pos * tr_neg - fl_pos * fl_neg) / \
        np.sqrt((tr_pos + fl_pos) * (tr_pos + fl_neg) *
                (tr_neg + fl_pos) * (tr_neg + fl_neg))
    return mcc


def measure_precision(conf_matrix):
    """
    """
    tr_pos, fl_pos, _, _ = conf_matrix.ravel()
    return tr_pos / (tr_pos + fl_pos)


def measure_recall(conf_matrix):
    """
    """
    tr_pos, _, fl_neg, _ = conf_matrix.ravel()
    return tr_pos / (tr_pos + fl_neg)


def measure_specificity(conf_matrix):
    """
    """
    _, fl_pos, _, tr_neg = conf_matrix.ravel()
    return tr_neg / (tr_neg + fl_pos)
