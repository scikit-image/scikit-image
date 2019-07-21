
import numpy as np
from ..metrics.simple_metrics import (mean_squared_error,
                                       peak_signal_noise_ratio,
                                       normalized_root_mse)
from .._shared.utils import warn

__all__ = ['compare_mse',
           'compare_nrmse',
           'compare_psnr',
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
