
import numpy as np
from scipy.stats import entropy
from ..metrics.simple_metrics import (mean_squared_error,
                                       peak_signal_noise_ratio,
                                       normalized_root_mse)
from .._shared.utils import warn

__all__ = ['compare_mse',
           'compare_nrmse',
           'compare_psnr',
           'compare_nmi',
           ]


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
    im_true : ndarray
        Ground-truth image, same shape as im_test.
    im_test : ndarray
        Test image.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    """
    warn('DEPRECATED: skimage.measure.compare_mse has been moved to '
         'skimage.metrics.mean_squared_error. It will be removed from '
         'skimage.measure in version 0.18.')
    return mean_squared_error(im1, im2)


def compare_nrmse(im_true, im_test, norm_type='euclidean'):
    """Compute the normalized root mean-squared error (NRMSE) between two
    images.

    Parameters
    ----------
    im_true : ndarray
        Ground-truth image, same shape as im_test.
    im_test : ndarray
        Test image.
    norm_type : {'euclidean', 'min-max', 'mean'}
        Controls the normalization method to use in the denominator of the
        NRMSE.  There is no standard method of normalization across the
        literature [1]_.  The methods available here are as follows:

        - 'euclidean' : normalize by the averaged Euclidean norm of
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
    warn('DEPRECATED: skimage.measure.compare_nrmse has been moved to '
         'skimage.metrics.normalized_root_mse. It will be removed from '
         'skimage.measure in version 0.18.')
    return normalized_root_mse(im_true, im_test, norm_type=norm_type)


def compare_psnr(im_true, im_test, data_range=None):
    """ Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    im_true : ndarray
        Ground-truth image, same shape as im_test.
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
    warn('DEPRECATED: skimage.measure.compare_psnr has been moved to '
         'skimage.metrics.peak_signal_noise_ratio. It will be removed from '
         'skimage.measure in version 0.18.')
    return peak_signal_noise_ratio(im_true, im_test, data_range=data_range)


def _pad_to(arr, shape):
    """Pad an array with trailing zeros to a given target shape.

    Parameters
    ----------
    arr : ndarray
        The input array.
    shape : tuple
        The target shape.

    Returns
    -------
    padded : ndarray
        The padded array.

    Examples
    --------
    >>> _pad_to(np.ones((1, 1), dtype=int), (1, 3))
    array([[1, 0, 0]])
    """
    if not all(s >= i for s, i in zip(shape, arr.shape)):
        raise ValueError("Target shape must be strictly greater "
                         "than input shape.")
    padding = [(0, s-i) for s, i in zip(shape, arr.shape)]
    return np.pad(arr, pad_width=padding, mode='constant', constant_values=0)


def compare_nmi(im_true, im_test, *, bins=100):
    """Compute the normalized mutual information.

    The normalized mutual information is given by::

    ..math::

        Y(A, B) = \frac{H(A) + H(B)}{H(A, B)}

    where :math:`H(X)` is the entropy,
    :math:`- \sum_{x \in X}{x \log x}.`

    It was proposed to be useful in registering images by Colin Studholme and
    colleagues [1]_. It ranges from 1 (perfectly uncorrelated image values)
    to 2 (perfectly correlated image values, whether positively or negatively).

    Parameters
    ----------
    im_true, im_test : ndarray
        Images to be compared. The two input images must have the same number
        of dimensions.
    bins : int or sequence of int, optional
        The number of bins along each axis of the joint histogram.

    Returns
    -------
    nmi : float
        The normalized mutual information between the two arrays, computed at
        the granularity given by ``bins``. Higher NMI implies more similar
        input images.

    Raises
    ------
    ValueError
        If the images don't have the same number of dimensions.

    Notes
    -----
    If the two input images are not the same shape, the smaller image is padded
    with zeros.

    References
    ----------
    .. [1] C. Studholme, D.L.G. Hill, & D.J. Hawkes (1999). An overlap
           invariant entropy measure of 3D medical image alignment.
           Pattern Recognition 32(1):71-86
           :DOI:`10.1016/S0031-3203(98)00091-0`
    """
    if im_true.ndim != im_test.ndim:
        raise ValueError('NMI requires images of same number of dimensions. '
                         'Got {}D for `im_true` and {}D for `im_test`.'
                         .format(im_true.ndim, im_test.ndim))
    if im_true.shape != im_test.shape:
        max_shape = np.maximum(im_true.shape, im_test.shape)
        padded_true = _pad_to(im_true, max_shape)
        padded_test = _pad_to(im_test, max_shape)
    else:
        padded_true, padded_test = im_true, im_test

    hist, bin_edges = np.histogramdd([np.ravel(padded_true),
                                      np.ravel(padded_test)],
                                     bins=bins, density=True)

    H_im_true = entropy(np.sum(hist, axis=0))
    H_im_test = entropy(np.sum(hist, axis=1))
    H_true_test = entropy(np.ravel(hist))

    return (H_im_true + H_im_test) / H_true_test
