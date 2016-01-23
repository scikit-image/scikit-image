from __future__ import division

import numpy as np
from ..util.dtype import dtype_range

__all__ = ['mse', 'nrmse', 'psnr']


def mse(X, Y):
    """ compute mean-squared error between two images.

    Parameters
    ----------
    X, Y : ndarray
        Image.  Any dimensionality.

    Returns
    -------
    mse : float
        The MSE metric.

    """
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')
    if not X.dtype == Y.dtype:
        raise ValueError('Input images must have the same dtype.')
    return np.square(X - Y).mean()


def nrmse(im_true, im, norm_type='Euclidean'):
    """ compute the normalized root mean-squared error between two images.

    Parameters
    ----------
    im_true : ndarray
        Ground-truth image.
    im : ndarray
        Test image.
    norm_type : {'Euclidean', 'min-max', 'mean'}
        Controls the normalization method to use in the denominator of the
        NRMSE.

    Returns
    -------
    nrmse : float
        The NRMSE metric.

    Notes
    -----
    There is no standard method of normalization across the literature [1]_.
    The methods available here are as follows:

    - 'Euclidean' : normalize by the Euclidean norm of ``im_true``.
    - 'min-max'   : normalize by the intensity range of ``im_true``.
    - 'mean'      : normalize by the mean of ``im_true``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Root-mean-square_deviation

    """
    if not im_true.dtype == im.dtype:
        raise ValueError('Input images must have the same dtype.')

    if not im_true.shape == im.shape:
        raise ValueError('Input images must have the same dimensions.')

    norm_type = norm_type.lower()
    if norm_type == 'euclidean':
        denom = np.sqrt((im_true*im_true).mean())
    elif norm_type == 'min-max':
        denom = im_true.max() - im_true.min()
    elif norm_type == 'mean':
        denom = im_true.mean()
    return np.sqrt(mse(im_true, im)) / denom


def psnr(im_true, im, dynamic_range=None):
    """ Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    im_true : ndarray
        Ground-truth image.
    im : ndarray
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
    if not im_true.dtype == im.dtype:
        raise ValueError('Input images must have the same dtype.')

    if not im_true.shape == im.shape:
        raise ValueError('Input images must have the same dimensions.')

    if dynamic_range is None:
        dmin, dmax = dtype_range[im_true.dtype.type]
        dynamic_range = dmax - dmin

    im_true = im_true.astype(np.float64)
    im = im.astype(np.float64)

    err = mse(im_true, im)
    psnr = 10 * np.log10((dynamic_range ** 2) / err)
    return psnr
