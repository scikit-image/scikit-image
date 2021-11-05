"""Filters used across multiple skimage submodules.

These are defined here to avoid circular imports.

The unit tests remain under skimage/filters/tests/
"""
from collections.abc import Iterable

import numpy as np
from scipy import ndimage as ndi

from .._shared import utils
from .._shared.utils import _supported_float_type, convert_to_float, warn


SOBEL_EDGE = np.array([1, 0, -1])
SOBEL_SMOOTH = np.array([1, 2, 1]) / 4


@utils.deprecate_multichannel_kwarg(multichannel_position=5)
def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
             multichannel=None, preserve_range=False, truncate=4.0, *,
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
    multichannel : bool, optional (default: None)
        Whether the last axis of the image is to be interpreted as multiple
        channels. If True, each channel is filtered separately (channels are
        not mixed together). Only 3 channels are supported. If ``None``,
        the function will attempt to guess this, and raise a warning if
        ambiguous, when the array has shape (M, N, 3).
        This argument is deprecated: specify `channel_axis` instead.
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
    if image.ndim == 3 and image.shape[-1] == 3 and channel_axis is None:
        msg = ("Images with dimensions (M, N, 3) are interpreted as 2D+RGB "
               "by default. Use `multichannel=False` to interpret as "
               "3D image with last dimension of length 3.")
        warn(RuntimeWarning(msg))
        channel_axis = -1
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


def _kernel_shape(ndim, dim):
    """Return list of `ndim` 1s except at position `dim`, where value is -1.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the kernel shape.
    dim : int
        The axis of the kernel to expand to shape -1.

    Returns
    -------
    shape : list of int
        The requested shape.

    Examples
    --------
    >>> _kernel_shape(2, 0)
    [-1, 1]
    >>> _kernel_shape(3, 1)
    [1, -1, 1]
    >>> _kernel_shape(4, -1)
    [1, 1, 1, -1]
    """
    shape = [1, ] * ndim
    shape[dim] = -1
    return shape


def _reshape_nd(arr, ndim, dim):
    """Reshape a 1D array to have n dimensions, all singletons but one.

    Parameters
    ----------
    arr : array, shape (N,)
        Input array
    ndim : int
        Number of desired dimensions of reshaped array.
    dim : int
        Which dimension/axis will not be singleton-sized.

    Returns
    -------
    arr_reshaped : array, shape ([1, ...], N, [1,...])
        View of `arr` reshaped to the desired shape.

    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> arr = rng.random(7)
    >>> _reshape_nd(arr, 2, 0).shape
    (7, 1)
    >>> _reshape_nd(arr, 3, 1).shape
    (1, 7, 1)
    >>> _reshape_nd(arr, 4, -1).shape
    (1, 1, 1, 7)
    """
    kernel_shape = _kernel_shape(ndim, dim)
    return np.reshape(arr, kernel_shape)


def _mask_filter_result(result, mask):
    """Return result after masking.

    Input masks are eroded so that mask areas in the original image don't
    affect values in the result.
    """
    if mask is not None:
        erosion_footprint = ndi.generate_binary_structure(mask.ndim, mask.ndim)
        mask = ndi.binary_erosion(mask, erosion_footprint, border_value=0)
        result *= mask
    return result


def _generic_edge_filter(image, *, smooth_weights, edge_weights=[1, 0, -1],
                         axis=None, mode='reflect', cval=0.0, mask=None):
    """Apply a generic, n-dimensional edge filter.

    The filter is computed by applying the edge weights along one dimension
    and the smoothing weights along all other dimensions. If no axis is given,
    or a tuple of axes is given the filter is computed along all axes in turn,
    and the magnitude is computed as the square root of the average square
    magnitude of all the axes.

    Parameters
    ----------
    image : array
        The input image.
    smooth_weights : array of float
        The smoothing weights for the filter. These are applied to dimensions
        orthogonal to the edge axis.
    edge_weights : 1D array of float, optional
        The weights to compute the edge along the chosen axes.
    axis : int or sequence of int, optional
        Compute the edge filter along this axis. If not provided, the edge
        magnitude is computed. This is defined as::

            edge_mag = np.sqrt(sum([_generic_edge_filter(image, ..., axis=i)**2
                                    for i in range(image.ndim)]) / image.ndim)

        The magnitude is also computed if axis is a sequence.
    mode : str or sequence of str, optional
        The boundary mode for the convolution. See `scipy.ndimage.convolve`
        for a description of the modes. This can be either a single boundary
        mode or one boundary mode per axis.
    cval : float, optional
        When `mode` is ``'constant'``, this is the constant used in values
        outside the boundary of the image data.
    """
    ndim = image.ndim
    if axis is None:
        axes = list(range(ndim))
    elif np.isscalar(axis):
        axes = [axis]
    else:
        axes = axis
    return_magnitude = (len(axes) > 1)

    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    output = np.zeros(image.shape, dtype=float_dtype)

    for edge_dim in axes:
        kernel = _reshape_nd(edge_weights, ndim, edge_dim)
        smooth_axes = list(set(range(ndim)) - {edge_dim})
        for smooth_dim in smooth_axes:
            kernel = kernel * _reshape_nd(smooth_weights, ndim, smooth_dim)
        ax_output = ndi.convolve(image, kernel, mode=mode)
        if return_magnitude:
            ax_output *= ax_output
        output += ax_output

    if return_magnitude:
        output = np.sqrt(output) / np.sqrt(ndim)
    return output


def sobel(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    """Find edges in an image using the Sobel filter.

    Parameters
    ----------
    image : array
        The input image.
    mask : array of bool, optional
        Clip the output image to this mask. (Values where mask=0 will be set
        to 0.)
    axis : int or sequence of int, optional
        Compute the edge filter along this axis. If not provided, the edge
        magnitude is computed. This is defined as::

            sobel_mag = np.sqrt(sum([sobel(image, axis=i)**2
                                     for i in range(image.ndim)]) / image.ndim)

        The magnitude is also computed if axis is a sequence.
    mode : str or sequence of str, optional
        The boundary mode for the convolution. See `scipy.ndimage.convolve`
        for a description of the modes. This can be either a single boundary
        mode or one boundary mode per axis.
    cval : float, optional
        When `mode` is ``'constant'``, this is the constant used in values
        outside the boundary of the image data.

    Returns
    -------
    output : array of float
        The Sobel edge map.

    See also
    --------
    sobel_h, sobel_v : horizontal and vertical edge detection.
    scharr, prewitt, farid, skimage.feature.canny

    References
    ----------
    .. [1] D. Kroon, 2009, Short Paper University Twente, Numerical
           Optimization of Kernel Based Image Derivatives.

    .. [2] https://en.wikipedia.org/wiki/Sobel_operator

    Examples
    --------
    >>> from skimage import data
    >>> from skimage import filters
    >>> camera = data.camera()
    >>> edges = filters.sobel(camera)
    """
    output = _generic_edge_filter(image, smooth_weights=SOBEL_SMOOTH,
                                  axis=axis, mode=mode, cval=cval)
    output = _mask_filter_result(output, mask)
    return output
