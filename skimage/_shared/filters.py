"""Filters used across multiple skimage submodules.

These are defined here to avoid circular imports.

The unit tests remain under skimage/filters/tests/
"""
from collections.abc import Iterable

import numpy as np
from scipy import ndimage as ndi

from .._shared.utils import _supported_float_type, convert_to_float


def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
             preserve_range=False, truncate=4.0, *,
             channel_axis=None):
    """Multi-dimensional Gaussian filter.

    Parameters
    ----------
    image : array-like
        Input image (grayscale or color) to filter.
    sigma : scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    output : array, optional
        The ``output`` parameter passes an array in which to store the
        filter output.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'nearest'.
    cval : scalar, optional
        Value to fill past edges of input if ``mode`` is 'constant'. Default
        is 0.0
    preserve_range : bool, optional
        If True, keep the original range of values. Otherwise, the input
        ``image`` is converted according to the conventions of ``img_as_float``
        (Normalized first to values [-1.0 ; 1.0] or [0 ; 1.0] depending on
        dtype of input)

        For more information, see:
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    truncate : float, optional
        Truncate the filter at this many standard deviations.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

        .. warning::

            Automatic detection of the color channel based on the old deprecated
            `multichannel=None` was broken in version 0.19. In 0.20 this
            behavior is fixed. The last axis of an `image` with dimensions
            (M, N, 3) is interpreted as a color channel if `channel_axis` is not
            set by the user (signaled by the default proxy value
            `ChannelAxisNotSet`). Starting with 0.22, `channel_axis=None` will
            be used as the new default value.

    Returns
    -------
    filtered_image : ndarray
        the filtered array

    Notes
    -----
    This function is a wrapper around :func:`scipy.ndi.gaussian_filter`.

    Integer arrays are converted to float.

    The ``output`` should be floating point data type since gaussian converts
    to float provided ``image``. If ``output`` is not provided, another array
    will be allocated and returned as the result.

    The multi-dimensional filter is implemented as a sequence of
    one-dimensional convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.

    Examples
    --------

    >>> a = np.zeros((3, 3))
    >>> a[1, 1] = 1
    >>> a
    array([[0., 0., 0.],
           [0., 1., 0.],
           [0., 0., 0.]])
    >>> gaussian(a, sigma=0.4)  # mild smoothing
    array([[0.00163116, 0.03712502, 0.00163116],
           [0.03712502, 0.84496158, 0.03712502],
           [0.00163116, 0.03712502, 0.00163116]])
    >>> gaussian(a, sigma=1)  # more smoothing
    array([[0.05855018, 0.09653293, 0.05855018],
           [0.09653293, 0.15915589, 0.09653293],
           [0.05855018, 0.09653293, 0.05855018]])
    >>> # Several modes are possible for handling boundaries
    >>> gaussian(a, sigma=1, mode='reflect')
    array([[0.08767308, 0.12075024, 0.08767308],
           [0.12075024, 0.16630671, 0.12075024],
           [0.08767308, 0.12075024, 0.08767308]])
    >>> # For RGB images, each is filtered separately
    >>> from skimage.data import astronaut
    >>> image = astronaut()
    >>> filtered_img = gaussian(image, sigma=1, channel_axis=-1)

    """
    if np.any(np.asarray(sigma) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")
    if channel_axis is not None:
        # do not filter across channels
        if not isinstance(sigma, Iterable):
            sigma = [sigma] * (image.ndim - 1)
        if len(sigma) == image.ndim - 1:
            sigma = list(sigma)
            sigma.insert(channel_axis % image.ndim, 0)
    image = convert_to_float(image, preserve_range)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if (output is not None) and (not np.issubdtype(output.dtype, np.floating)):
        raise ValueError("Provided output data type is not float")
    return ndi.gaussian_filter(image, sigma, output=output,
                               mode=mode, cval=cval, truncate=truncate)
