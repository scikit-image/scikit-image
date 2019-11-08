import numpy as np
from ..util.dtype import dtype_range
from .._shared.utils import skimage_deprecation, warn
from ..util import img_as_float
from ..exposure import rescale_intensity
from scipy.ndimage import maximum_filter, minimum_filter
from warnings import warn
import numpy as np
from ..metrics.simple_metrics import (mean_squared_error,
                                      peak_signal_noise_ratio,
                                      normalized_root_mse)

__all__ = ['compare_mse',
           'compare_nrmse',
           'compare_psnr',
           'enhancement_measure'
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


def enhancement_measure(image: np.ndarray,
                        size: int = 3,
                        eps: float = 1e-6) -> float:
    """ The image enhancement measure called EME based on [1]_.
        It is a way of quantifying improvement of the image after enhancement.

        The function uses a sliding window of user-provided size to measure
        the mean of log of maximal and minimal intensity ratio
        within the window.

        Parameters
        ----------
        image : array
            Input image of which the quality should be assessed.
            Can be either 3-channel RGB or 1-channel grayscale.
        size : int, optional
            Size of the window.
        eps : float, optional
            Parameter to avoid division by zero.

        Returns
        -------
        eme : float
            The number describing image quality.

        References
        ----------
        .. [1] Sos S. Agaian,  Karen Panetta, and Artyom M. Grigoryan.
               "A new measure of image enhancement.",
               IASTED International Conference on Signal Processing
               & Communication, Citeseer, 2000,
               :DOI:10.1.1.35.4021

        Examples
        --------
        >>> from skimage.data import camera
        >>> from skimage.exposure import equalize_hist
        >>> img = camera()
        >>> before = enhancement_measure(img)
        >>> after = enhancement_measure(equalize_hist(img))
        >>> before < after
        True
    """
    image = img_as_float(image)
    image = rescale_intensity(image, out_range=(0., 1.))
    eme = np.divide(maximum_filter(image, size=size),
                    minimum_filter(image, size=size) + eps)
    eme = np.mean(20 * np.log(eme + eps))
    return eme
