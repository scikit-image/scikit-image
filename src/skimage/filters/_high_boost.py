"""High-boost sharpening filter."""

import numpy as np
from scipy.ndimage import gaussian_filter

from skimage._shared.utils import _supported_float_type


def _gaussian_sigma(ndim, sigma, channel_axis):
    """Return Gaussian sigmas, with no smoothing along ``channel_axis``."""
    if channel_axis is None:
        return sigma
    axis = channel_axis % ndim
    sigmas = [sigma] * ndim
    sigmas[axis] = 0.0
    return sigmas


def high_boost_filter(image, boost_factor=1.0, sigma=1.0, *, channel_axis=None):
    """Sharpen an image with a high-boost filter.

    Edges are emphasized by adding an amplified residual back to the image:

    ``output = image + boost_factor * (image - smoothed)``

    ``smoothed`` is computed with a Gaussian filter. Filtering uses a copy of
    ``image``, so the input array is left unchanged.

    Parameters
    ----------
    image : ndarray
        Input image. Real dtypes are converted with
        ``_supported_float_type``: ``float32`` is preserved, ``float16`` is
        promoted to ``float32``, and other real dtypes are converted to
        ``float64``.
    boost_factor : float, optional
        Amplification of the high-frequency residual ``image - smoothed``.
        Must be greater than or equal to 0. A value of ``0`` returns a
        floating-point copy of ``image``.
    sigma : float, optional
        Standard deviation of the Gaussian kernel used for ``smoothed``.
        Must be greater than or equal to 0. The same value is used for every
        spatial axis.
    channel_axis : int or None, optional
        Axis that indexes channels. ``None`` filters every axis. Otherwise
        that axis is not smoothed.

    Returns
    -------
    output : ndarray
        Sharpened image. Shape matches ``image``. Dtype is the supported
        floating-point dtype of ``image``.

    Raises
    ------
    ValueError
        If ``boost_factor`` is negative, if ``sigma`` is negative, or if
        ``image`` is complex-valued.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.filters import high_boost_filter
    >>> image = np.zeros((3, 3))
    >>> image[1, 1] = 1
    >>> high_boost_filter(image, boost_factor=1, sigma=1)
    array([[-0.05855018, -0.09653293, -0.05855018],
           [-0.09653293,  1.84084411, -0.09653293],
           [-0.05855018, -0.09653293, -0.05855018]])
    """
    if boost_factor < 0:
        raise ValueError("boost_factor must be greater than or equal to 0.")
    if sigma < 0:
        raise ValueError("sigma must be greater than or equal to 0.")

    image = np.asarray(image)
    float_dtype = _supported_float_type(image.dtype)
    # Copy before filtering so the caller's array is never modified.
    working = np.array(image, dtype=float_dtype, copy=True)
    sigmas = _gaussian_sigma(working.ndim, float(sigma), channel_axis)
    smoothed = gaussian_filter(working, sigma=sigmas, mode="nearest")
    factor = np.array(boost_factor, dtype=float_dtype)
    # inf - inf is NaN and warns; non-finite inputs stay non-finite.
    with np.errstate(invalid="ignore"):
        output = working + factor * (working - smoothed)
    return output
