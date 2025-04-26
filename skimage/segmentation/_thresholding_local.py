"""Local thresholding.

These algorithms create locally variying thresholds for each pixel in an image.
"""

import itertools
import math
from collections.abc import Iterable

import numpy as np
from scipy import ndimage as ndi

from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type
from ..transform import integral_image
from ..util import dtype_limits
from ..filters._sparse import _correlate_sparse, _validate_window_size


def _mean_std(image, w):
    """Return local mean and standard deviation of each pixel using a
    neighborhood defined by a rectangular window size ``w``.
    The algorithm uses integral images to speedup computation. This is
    used by :func:`threshold_niblack` and :func:`threshold_sauvola`.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    w : int, or iterable of int
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).

    Returns
    -------
    m : ndarray of float, same shape as ``image``
        Local mean of the image.
    s : ndarray of float, same shape as ``image``
        Local standard deviation of the image.

    References
    ----------
    .. [1] F. Shafait, D. Keysers, and T. M. Breuel, "Efficient
           implementation of local adaptive thresholding techniques
           using integral images." in Document Recognition and
           Retrieval XV, (San Jose, USA), Jan. 2008.
           :DOI:`10.1117/12.767755`
    """

    if not isinstance(w, Iterable):
        w = (w,) * image.ndim
    _validate_window_size(w)

    float_dtype = _supported_float_type(image.dtype)
    pad_width = tuple((k // 2 + 1, k // 2) for k in w)
    padded = np.pad(image.astype(float_dtype, copy=False), pad_width, mode='reflect')

    # Note: keep float64 integral images for accuracy. Outputs of
    # _correlate_sparse can later be safely cast to float_dtype
    integral = integral_image(padded, dtype=np.float64)
    padded *= padded
    integral_sq = integral_image(padded, dtype=np.float64)

    # Create lists of non-zero kernel indices and values
    kernel_indices = list(itertools.product(*tuple([(0, _w) for _w in w])))
    kernel_values = [
        (-1) ** (image.ndim % 2 != np.sum(indices) % 2) for indices in kernel_indices
    ]

    total_window_size = math.prod(w)
    kernel_shape = tuple(_w + 1 for _w in w)
    m = _correlate_sparse(integral, kernel_shape, kernel_indices, kernel_values)
    m = m.astype(float_dtype, copy=False)
    m /= total_window_size
    g2 = _correlate_sparse(integral_sq, kernel_shape, kernel_indices, kernel_values)
    g2 = g2.astype(float_dtype, copy=False)
    g2 /= total_window_size
    # Note: we use np.clip because g2 is not guaranteed to be greater than
    # m*m when floating point error is considered
    s = np.sqrt(np.clip(g2 - m * m, 0, None))
    return m, s


def threshold_local(
    image,
    *,
    block_size=3,
    method='gaussian',
    offset=0,
    mode='reflect',
    param=None,
    cval=0,
):
    """Compute a threshold mask image based on local pixel neighborhood.

    Also known as adaptive or dynamic thresholding. The threshold value is
    the weighted mean for the local neighborhood of a pixel subtracted by a
    constant. Alternatively the threshold can be determined dynamically by a
    given function, using the 'generic' method.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    block_size : int or sequence of int
        Odd size of pixel neighborhood which is used to calculate the
        threshold value (e.g. 3, 5, 7, ..., 21, ...).
    method : {'generic', 'gaussian', 'mean', 'median'}, optional
        Method used to determine adaptive threshold for local neighborhood in
        weighted mean image.

        * 'generic': use custom function (see ``param`` parameter)
        * 'gaussian': apply gaussian filter (see ``param`` parameter for custom\
                      sigma value)
        * 'mean': apply arithmetic mean filter
        * 'median': apply median rank filter

        By default, the 'gaussian' method is used.
    offset : float, optional
        Constant subtracted from weighted mean of neighborhood to calculate
        the local threshold value. Default offset is 0.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
        Default is 'reflect'.
    param : {int, function}, optional
        Either specify sigma for 'gaussian' method or function object for
        'generic' method. This functions takes the flat array of local
        neighborhood as a single argument and returns the calculated
        threshold for the centre pixel.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    threshold : (M, N[, ...]) ndarray
        Threshold image. All pixels in the input image higher than the
        corresponding pixel in the threshold image are considered foreground.

    References
    ----------
    .. [1] Gonzalez, R. C. and Wood, R. E. "Digital Image Processing
           (2nd Edition)." Prentice-Hall Inc., 2002: 600--612.
           ISBN: 0-201-18075-8

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()[:50, :50]
    >>> binary_image1 = image > threshold_local(
    ...     image, block_size=15, method='mean'
    ... )
    >>> func = lambda arr: arr.mean()
    >>> binary_image2 = image > threshold_local(
    ...     image, block_size=15, method='generic', param=func
    ... )

    """

    if np.isscalar(block_size):
        block_size = (block_size,) * image.ndim
    elif len(block_size) != image.ndim:
        raise ValueError("len(block_size) must equal image.ndim.")
    block_size = tuple(block_size)
    if any(b % 2 == 0 for b in block_size):
        raise ValueError(
            f'block_size must be odd! Given block_size '
            f'{block_size} contains even values.'
        )
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    thresh_image = np.zeros(image.shape, dtype=float_dtype)
    if method == 'generic':
        ndi.generic_filter(
            image, param, block_size, output=thresh_image, mode=mode, cval=cval
        )
    elif method == 'gaussian':
        if param is None:
            # automatically determine sigma which covers > 99% of distribution
            sigma = tuple([(b - 1) / 6.0 for b in block_size])
        else:
            sigma = param
        gaussian(image, sigma=sigma, out=thresh_image, mode=mode, cval=cval)
    elif method == 'mean':
        ndi.uniform_filter(image, block_size, output=thresh_image, mode=mode, cval=cval)
    elif method == 'median':
        ndi.median_filter(image, block_size, output=thresh_image, mode=mode, cval=cval)
    else:
        raise ValueError(
            "Invalid method specified. Please use `generic`, "
            "`gaussian`, `mean`, or `median`."
        )

    return thresh_image - offset


def threshold_local_niblack(image, *, window_size=15, k=0.2):
    """Applies Niblack local threshold to an array.

    A threshold T is calculated for every pixel in the image using the
    following formula::

        T = m(x,y) - k * s(x,y)

    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by a rectangular window with size w
    times w centered around the pixel. k is a configurable parameter
    that weights the effect of standard deviation.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    window_size : int, or iterable of int, optional
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).
    k : float, optional
        Value of parameter k in threshold formula.

    Returns
    -------
    threshold : (M, N[, ...]) ndarray
        Threshold mask. All pixels with an intensity higher than
        this value are assumed to be foreground.

    Notes
    -----
    This algorithm is originally designed for text recognition.

    The Bradley threshold is a particular case of the Niblack
    one, being equivalent to

    >>> from skimage import data
    >>> image = data.page()
    >>> q = 1
    >>> threshold_image = threshold_local_niblack(image, k=0) * q

    for some value ``q``. By default, Bradley and Roth use ``q=1``.


    References
    ----------
    .. [1] W. Niblack, An introduction to Digital Image Processing,
           Prentice-Hall, 1986.
    .. [2] D. Bradley and G. Roth, "Adaptive thresholding using Integral
           Image", Journal of Graphics Tools 12(2), pp. 13-21, 2007.
           :DOI:`10.1080/2151237X.2007.10129236`

    Examples
    --------
    >>> from skimage import data
    >>> image = data.page()
    >>> threshold_image = threshold_local_niblack(image, window_size=7, k=0.1)
    """
    m, s = _mean_std(image, window_size)
    return m - k * s


def threshold_local_sauvola(image, *, window_size=15, k=0.2, r=None):
    """Applies Sauvola local threshold to an array. Sauvola is a
    modification of Niblack technique.

    In the original method a threshold T is calculated for every pixel
    in the image using the following formula::

        T = m(x,y) * (1 + k * ((s(x,y) / R) - 1))

    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by a rectangular window with size w
    times w centered around the pixel. k is a configurable parameter
    that weights the effect of standard deviation.
    R is the maximum standard deviation of a grayscale image.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    window_size : int, or iterable of int, optional
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).
    k : float, optional
        Value of the positive parameter k.
    r : float, optional
        Value of R, the dynamic range of standard deviation.
        If None, set to the half of the image dtype range.

    Returns
    -------
    threshold : (M, N[, ...]) ndarray
        Threshold mask. All pixels with an intensity higher than
        this value are assumed to be foreground.

    Notes
    -----
    This algorithm is originally designed for text recognition.

    References
    ----------
    .. [1] J. Sauvola and M. Pietikainen, "Adaptive document image
           binarization," Pattern Recognition 33(2),
           pp. 225-236, 2000.
           :DOI:`10.1016/S0031-3203(99)00055-2`

    Examples
    --------
    >>> from skimage import data
    >>> image = data.page()
    >>> t_sauvola = threshold_local_sauvola(image, window_size=15, k=0.2)
    >>> binary_image = image > t_sauvola
    """
    if r is None:
        imin, imax = dtype_limits(image, clip_negative=False)
        r = 0.5 * (imax - imin)
    m, s = _mean_std(image, window_size)
    return m * (1 + k * ((s / r) - 1))


def threshold_labels_hysteresis(image, *, low, high):
    """Apply hysteresis thresholding to ``image``.

    This algorithm finds regions where ``image`` is greater than ``high``
    OR ``image`` is greater than ``low`` *and* that region is connected to
    a region greater than ``high``.

    Parameters
    ----------
    image : (M[, ...]) ndarray
        Grayscale input image.
    low : float, or array of same shape as ``image``
        Lower threshold.
    high : float, or array of same shape as ``image``
        Higher threshold.

    Returns
    -------
    thresholded : (M[, ...]) array of bool
        Array in which ``True`` indicates the locations where ``image``
        was above the hysteresis threshold.

    Examples
    --------
    >>> image = np.array([1, 2, 3, 2, 1, 2, 1, 3, 2])
    >>> threshold_labels_hysteresis(image, low=1.5, high=2.5).astype(int)
    array([0, 1, 1, 1, 0, 0, 0, 1, 1])

    References
    ----------
    .. [1] J. Canny. A computational approach to edge detection.
           IEEE Transactions on Pattern Analysis and Machine Intelligence.
           1986; vol. 8, pp.679-698.
           :DOI:`10.1109/TPAMI.1986.4767851`
    """
    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
    mask_low = image > low
    mask_high = image > high
    # Connected components of mask_low
    labels_low, num_labels = ndi.label(mask_low)
    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded
