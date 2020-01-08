from warnings import warn
import numpy as np
from ..metrics.simple_metrics import (mean_squared_error,
                                       peak_signal_noise_ratio,
                                       normalized_root_mse)

__all__ = ['compare_mse',
           'compare_nrmse',
           'compare_psnr',
           ]


def compare_mse(im1, im2):
    warn('DEPRECATED: skimage.measure.compare_mse has been moved to '
         'skimage.metrics.mean_squared_error. It will be removed from '
         'skimage.measure in version 0.18.', stacklevel=2)
    return mean_squared_error(im1, im2)


if mean_squared_error.__doc__ is not None:
    compare_mse.__doc__ = mean_squared_error.__doc__ + """
    Warns
    -----
    Deprecated:
        .. versionadded:: 0.16

        This function is deprecated and will be removed in scikit-image 0.18.
        Please use the function named ``mean_squared_error`` from the
        ``metrics`` module instead.

    See also
    --------
    skimage.metrics.mean_squared_error
    """


def compare_nrmse(im_true, im_test, norm_type='euclidean'):
    warn('DEPRECATED: skimage.measure.compare_nrmse has been moved to '
         'skimage.metrics.normalized_root_mse. It will be removed from '
         'skimage.measure in version 0.18.', stacklevel=2)
    return normalized_root_mse(im_true, im_test, normalization=norm_type)


if normalized_root_mse.__doc__ is not None:
    compare_nrmse.__doc__ = normalized_root_mse.__doc__ + """
    Warns
    -----
    Deprecated:
        .. versionadded:: 0.16

        This function is deprecated and will be removed in scikit-image 0.18.
        Please use the function named ``normalized_root_mse`` from the
        ``metrics`` module instead.

    See also
    --------
    skimage.metrics.normalized_root_mse
    """


def compare_psnr(im_true, im_test, data_range=None):
    warn('DEPRECATED: skimage.measure.compare_psnr has been moved to '
         'skimage.metrics.peak_signal_noise_ratio. It will be removed from '
         'skimage.measure in version 0.18.', stacklevel=2)
    return peak_signal_noise_ratio(im_true, im_test, data_range=data_range)


if peak_signal_noise_ratio.__doc__ is not None:
    compare_psnr.__doc__ = peak_signal_noise_ratio.__doc__ + """
    Warns
    -----
    Deprecated:
        .. versionadded:: 0.16

        This function is deprecated and will be removed in scikit-image 0.18.
        Please use the function named ``peak_signal_noise_ratio`` from the
        ``metrics`` module instead.

    See also
    --------
    skimage.metrics.peak_signal_noise_ratio
    """
