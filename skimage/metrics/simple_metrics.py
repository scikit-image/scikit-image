import numpy as np
from ..util.dtype import dtype_range
from .._shared.utils import warn, check_shape_equality

__all__ = ['mean_squared_error',
           'normalized_root_mse',
           'peak_signal_noise_ratio',
           ]


def _as_floats(im1, im2):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2


def mean_squared_error(im1, im2):
    """
    Compute the mean-squared error between two images.

    Parameters
    ----------
    im1, im2 : ndarray
        Images.  Any dimensionality, must have same shape.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    """
    check_shape_equality(im1, im2)
    im1, im2 = _as_floats(im1, im2)
    return np.mean((im1 - im2) ** 2, dtype=np.float64)


def normalized_root_mse(im_true, im_test, norm_type='euclidean'):
    """
    Compute the normalized root mean-squared error (NRMSE) between two
    images.

    Parameters
    ----------
    im_true : ndarray
        Ground-truth image, same shape as im_test.
    im_test : ndarray
        Test image.
    norm_type : {'euclidean', 'min-max', 'mean'}, optional
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
    check_shape_equality(im_true, im_test)
    im_true, im_test = _as_floats(im_true, im_test)

    # Ensure that both 'Euclidean' and 'euclidean' match
    norm_type = norm_type.lower()
    if norm_type == 'euclidean':
        denom = np.sqrt(np.mean((im_true * im_true), dtype=np.float64))
    elif norm_type == 'min-max':
        denom = im_true.max() - im_true.min()
    elif norm_type == 'mean':
        denom = im_true.mean()
    else:
        raise ValueError("Unsupported norm_type")
    return np.sqrt(mean_squared_error(im_true, im_test)) / denom


def peak_signal_noise_ratio(im_true, im_test, data_range=None):
    """
    Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    im_true : ndarray
        Ground-truth image, same shape as im_test.
    im_test : ndarray
        Test image.
    data_range : int, optional
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
    check_shape_equality(im_true, im_test)

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

    err = mean_squared_error(im_true, im_test)
    return 10 * np.log10((data_range ** 2) / err)
