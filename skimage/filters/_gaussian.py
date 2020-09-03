from collections.abc import Iterable
import numpy as np
from scipy import ndimage as ndi

from ..util import img_as_float
from .._shared.utils import warn, convert_to_float


__all__ = ['gaussian', 'difference_of_gaussians']


def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
             multichannel=None, preserve_range=False, truncate=4.0):
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
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of ``img_as_float``.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    truncate : float, optional
        Truncate the filter at this many standard deviations.

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
    >>> filtered_img = gaussian(image, sigma=1, multichannel=True)

    """

    spatial_dims = None
    try:
        spatial_dims = _guess_spatial_dimensions(image)
    except ValueError:
        spatial_dims = image.ndim
    if spatial_dims is None and multichannel is None:
        msg = ("Images with dimensions (M, N, 3) are interpreted as 2D+RGB "
               "by default. Use `multichannel=False` to interpret as "
               "3D image with last dimension of length 3.")
        warn(RuntimeWarning(msg))
        multichannel = True
    if np.any(np.asarray(sigma) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")
    if multichannel:
        # do not filter across channels
        if not isinstance(sigma, Iterable):
            sigma = [sigma] * (image.ndim - 1)
        if len(sigma) != image.ndim:
            sigma = np.concatenate((np.asarray(sigma), [0]))
    image = convert_to_float(image, preserve_range)
    if output is None:
        output = np.empty_like(image)
    elif not np.issubdtype(output.dtype, np.floating):
        raise ValueError("Provided output data type is not float")
    ndi.gaussian_filter(image, sigma, output=output, mode=mode, cval=cval,
                        truncate=truncate)
    return output


def _guess_spatial_dimensions(image):
    """Make an educated guess about whether an image has a channels dimension.

    Parameters
    ----------
    image : ndarray
        The input image.

    Returns
    -------
    spatial_dims : int or None
        The number of spatial dimensions of ``image``. If ambiguous, the value
        is ``None``.

    Raises
    ------
    ValueError
        If the image array has less than two or more than four dimensions.
    """
    if image.ndim == 2:
        return 2
    if image.ndim == 3 and image.shape[-1] != 3:
        return 3
    if image.ndim == 3 and image.shape[-1] == 3:
        return None
    if image.ndim == 4 and image.shape[-1] == 3:
        return 3
    else:
        raise ValueError("Expected 2D, 3D, or 4D array, got %iD." % image.ndim)


def difference_of_gaussians(image, low_sigma, high_sigma=None, *,
                            mode='nearest', cval=0, multichannel=False,
                            truncate=4.0):
    """Find features between ``low_sigma`` and ``high_sigma`` in size.

    This function uses the Difference of Gaussians method for applying
    band-pass filters to multi-dimensional arrays. The input array is
    blurred with two Gaussian kernels of differing sigmas to produce two
    intermediate, filtered images. The more-blurred image is then subtracted
    from the less-blurred image. The final output image will therefore have
    had high-frequency components attenuated by the smaller-sigma Gaussian, and
    low frequency components will have been removed due to their presence in
    the more-blurred intermediate.

    Parameters
    ----------
    image : ndarray
        Input array to filter.
    low_sigma : scalar or sequence of scalars
        Standard deviation(s) for the Gaussian kernel with the smaller sigmas
        across all axes. The standard deviations are given for each axis as a
        sequence, or as a single number, in which case the single number is
        used as the standard deviation value for all axes.
    high_sigma : scalar or sequence of scalars, optional (default is None)
        Standard deviation(s) for the Gaussian kernel with the larger sigmas
        across all axes. The standard deviations are given for each axis as a
        sequence, or as a single number, in which case the single number is
        used as the standard deviation value for all axes. If None is given
        (default), sigmas for all axes are calculated as 1.6 * low_sigma.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'nearest'.
    cval : scalar, optional
        Value to fill past edges of input if ``mode`` is 'constant'. Default
        is 0.0
    multichannel : bool, optional (default: False)
        Whether the last axis of the image is to be interpreted as multiple
        channels. If True, each channel is filtered separately (channels are
        not mixed together).
    truncate : float, optional (default is 4.0)
        Truncate the filter at this many standard deviations.

    Returns
    -------
    filtered_image : ndarray
        the filtered array.

    See also
    --------
    skimage.feature.blog_dog

    Notes
    -----
    This function will subtract an array filtered with a Gaussian kernel
    with sigmas given by ``high_sigma`` from an array filtered with a
    Gaussian kernel with sigmas provided by ``low_sigma``. The values for
    ``high_sigma`` must always be greater than or equal to the corresponding
    values in ``low_sigma``, or a ``ValueError`` will be raised.

    When ``high_sigma`` is none, the values for ``high_sigma`` will be
    calculated as 1.6x the corresponding values in ``low_sigma``. This ratio
    was originally proposed by Marr and Hildreth (1980) [1]_ and is commonly
    used when approximating the inverted Laplacian of Gaussian, which is used
    in edge and blob detection.

    Input image is converted according to the conventions of ``img_as_float``.

    Except for sigma values, all parameters are used for both filters.

    Examples
    --------
    Apply a simple Difference of Gaussians filter to a color image:

    >>> from skimage.data import astronaut
    >>> from skimage.filters import difference_of_gaussians
    >>> filtered_image = difference_of_gaussians(astronaut(), 2, 10,
    ...                                          multichannel=True)

    Apply a Laplacian of Gaussian filter as approximated by the Difference
    of Gaussians filter:

    >>> filtered_image = difference_of_gaussians(astronaut(), 2,
    ...                                          multichannel=True)

    Apply a Difference of Gaussians filter to a grayscale image using different
    sigma values for each axis:

    >>> from skimage.data import camera
    >>> filtered_image = difference_of_gaussians(camera(), (2,5), (3,20))

    References
    ----------
    .. [1] Marr, D. and Hildreth, E. Theory of Edge Detection. Proc. R. Soc.
           Lond. Series B 207, 187-217 (1980).
           https://doi.org/10.1098/rspb.1980.0020

    """
    image = img_as_float(image)
    low_sigma = np.array(low_sigma, dtype='float', ndmin=1)
    if high_sigma is None:
        high_sigma = low_sigma * 1.6
    else:
        high_sigma = np.array(high_sigma, dtype='float', ndmin=1)

    if multichannel is True:
        spatial_dims = image.ndim - 1
    else:
        spatial_dims = image.ndim

    if len(low_sigma) != 1 and len(low_sigma) != spatial_dims:
        raise ValueError('low_sigma must have length equal to number of'
                         ' spatial dimensions of input')
    if len(high_sigma) != 1 and len(high_sigma) != spatial_dims:
        raise ValueError('high_sigma must have length equal to number of'
                         ' spatial dimensions of input')

    low_sigma = low_sigma * np.ones(spatial_dims)
    high_sigma = high_sigma * np.ones(spatial_dims)

    if any(high_sigma < low_sigma):
        raise ValueError('high_sigma must be equal to or larger than'
                         'low_sigma for all axes')

    im1 = gaussian(image, low_sigma, mode=mode, cval=cval,
                   multichannel=multichannel, truncate=truncate)

    im2 = gaussian(image, high_sigma, mode=mode, cval=cval,
                   multichannel=multichannel, truncate=truncate)

    return im1 - im2
