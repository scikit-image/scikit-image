from __future__ import division

import numpy as np
from ..util.dtype import dtype_range

__all__ = ['mse', 'nrmse', 'psnr']


def _as_floats(X, Y):
    """Promote X, Y to nearest appropriate floating point precision."""
    float_type = np.result_type(X.dtype, Y.dtype, np.float32)
    if X.dtype != float_type:
        X = X.astype(float_type)
    if Y.dtype != float_type:
        Y = Y.astype(float_type)
    return X, Y


def mse(X, Y):
    """Compute the mean-squared error between two images.

    Parameters
    ----------
    X, Y : ndarray
        Image.  Any dimensionality.

    Returns
    -------
    mse : float
        The MSE metric.

    """
    X, Y = _as_floats(X, Y)

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')
    if not X.dtype == Y.dtype:
        raise ValueError('Input images must have the same dtype.')
    return np.square(X - Y).mean()


def nrmse(im_true, im_test, norm_type='Euclidean'):
    """Compute the normalized root mean-squared error between two images.

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

        - 'Euclidean' : normalize by the Euclidean norm of ``im_true``.
        - 'min-max'   : normalize by the intensity range of ``im_true``.
        - 'mean'      : normalize by the mean of ``im_true``.

    Returns
    -------
    nrmse : float
        The NRMSE metric.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Root-mean-square_deviation

    """
    if not im_true.dtype == im_test.dtype:
        raise ValueError('Input images must have the same dtype.')

    if not im_true.shape == im_test.shape:
        raise ValueError('Input images must have the same dimensions.')

    im_true, im_test = _as_floats(im_true, im_test)

    norm_type = norm_type.lower()
    if norm_type == 'euclidean':
        denom = np.sqrt((im_true*im_true).mean())
    elif norm_type == 'min-max':
        denom = im_true.max() - im_true.min()
    elif norm_type == 'mean':
        denom = im_true.mean()
    else:
        raise ValueError("Unsupported norm_type")
    return np.sqrt(mse(im_true, im_test)) / denom


def psnr(im_true, im_test, dynamic_range=None):
    """ Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    im_true : ndarray
        Ground-truth image.
    im_test : ndarray
        Test image.
    dynamic_range : int
        The dynamic range of the input image (distance between minimum and
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
    if not im_true.dtype == im_test.dtype:
        raise ValueError('Input images must have the same dtype.')

    if not im_true.shape == im_test.shape:
        raise ValueError('Input images must have the same dimensions.')

    if dynamic_range is None:
        dmin, dmax = dtype_range[im_true.dtype.type]
        dynamic_range = dmax - dmin

    im_true, im_test = _as_floats(im_true, im_test)

    err = mse(im_true, im_test)
    return 10 * np.log10((dynamic_range ** 2) / err)
