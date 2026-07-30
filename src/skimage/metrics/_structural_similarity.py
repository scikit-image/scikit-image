import numpy as np

import _skimage2 as ski2

from _skimage2._shared._warnings import warn_external
from _skimage2.metrics._structural_similarity import __doc__  # noqa: F401

from .._migration import ski2_migration_decorator
from ..util.dtype import dtype_range


__all__ = ['structural_similarity']


@ski2_migration_decorator(
    r"""
``%(qname_old)s`` is deprecated in favor of
``%(qname_new)s``, which has a new signature.
The parameter `data_range` is now a required parameter.

If you didn't provide `data_range` explicitly before and relied on its
default behavior, you can keep the old (``skimage``, v1.x) behavior by
setting the parameter explicitly.

<!--- cond-start: doc -->
For example:

>>> import numpy as np
>>> import skimage as ski
>>> import skimage2 as ski2
...
>>> im1 = np.arange(20)
>>> im2 = im1 // 2
...
>>> result1 = ski.metrics.structural_similarity(im1, im2)
...
>>> data_range = np.iinfo(im1.dtype).max - np.iinfo(im1.dtype).min
>>> result2 = ski2.metrics.structural_similarity(im1, im2, data_range=data_range)
...
>>> np.testing.assert_equal(result1, result2)

<!--- cond-end -->
For floating dtypes, setting `data_range` was already required in
``skimage``/v1.x.
""",
    qname_old='skimage.metrics.structural_similarity',
)
def structural_similarity(
    im1,
    im2,
    *,
    win_size=None,
    gradient=False,
    data_range=None,
    channel_axis=None,
    gaussian_weights=False,
    full=False,
    **kwargs,
):
    """
    Compute the mean structural similarity index between two images.
    Please pay attention to the `data_range` parameter with floating-point images.

    Parameters
    ----------
    im1, im2 : ndarray
        Images. Any dimensionality with same shape.
    win_size : int or None, optional
        The side-length of the sliding window used in comparisons
        (default: 7). Must be an odd value. If `gaussian_weights` is
        True, `win_size` cannot be specified since the window size is
        determined by `sigma`.
    gradient : bool, optional
        If True, also return the gradient with respect to im2.
    data_range : float, optional
        The data range of the input image (difference between maximum and
        minimum possible values). By default, this is estimated from the image
        data type. This estimate may be wrong for floating-point image data.
        Therefore it is recommended to always pass this scalar value explicitly
        (see note below).
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    gaussian_weights : bool, optional
        If True, the local mean and variance are computed using a normalized
        Gaussian kernel of width `sigma` rather than a uniform window.
    full : bool, optional
        If True, also return the full structural similarity image.

    Other Parameters
    ----------------
    use_sample_covariance : bool
        If True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        Algorithm parameter, K1 (small constant, see [1]_).
    K2 : float
        Algorithm parameter, K2 (small constant, see [1]_).
    sigma : float
        Standard deviation for the Gaussian when `gaussian_weights` is True.
        Default is 1.5.

    Returns
    -------
    mssim : float
        The mean structural similarity index over the image.
    grad : ndarray
        The gradient of the structural similarity between im1 and im2 [2]_.
        This is only returned if `gradient` is set to True.
    S : ndarray
        The full SSIM image.  This is only returned if `full` is set to True.

    Notes
    -----
    If `data_range` is not specified, the range is automatically guessed
    based on the image data type. However for floating-point image data, this
    estimate yields a result double the value of the desired range, as the
    `dtype_range` in `skimage.util.dtype.py` has defined intervals from -1 to
    +1. This yields an estimate of 2, instead of 1, which is most often
    required when working with image data (as negative light intensities are
    nonsensical). In case of working with YCbCr-like color data, note that
    these ranges are different per channel (Cb and Cr have double the range
    of Y), so one cannot calculate a channel-averaged SSIM with a single call
    to this function, as identical ranges are assumed for each channel.

    To match the implementation of Wang et al. [1]_, set `gaussian_weights`
    to True, `sigma` to 1.5, `use_sample_covariance` to False, and
    specify the `data_range` argument.

    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_ssim`` to
        ``skimage.metrics.structural_similarity``.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       :DOI:`10.1109/TIP.2003.819861`

    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       :arxiv:`0901.0065`
       :DOI:`10.1007/s10043-009-0119-z`

    """
    if data_range is None:
        if np.issubdtype(im1.dtype, np.floating) or np.issubdtype(
            im2.dtype, np.floating
        ):
            raise ValueError(
                'Since image dtype is floating point, you must specify '
                'the data_range parameter. Please read the documentation '
                'carefully (including the note). It is recommended that '
                'you always specify the data_range anyway.'
            )
        if im1.dtype != im2.dtype:
            warn_external(
                "Inputs have mismatched dtypes. Setting data_range based on im1.dtype.",
            )
        dmin, dmax = dtype_range[im1.dtype.type]
        data_range = dmax - dmin
        if np.issubdtype(im1.dtype, np.integer) and (im1.dtype != np.uint8):
            warn_external(
                "Setting data_range based on im1.dtype. "
                + f"data_range = {data_range:.0f}. "
                + "Please specify data_range explicitly to avoid mistakes.",
            )

    if gaussian_weights and win_size is not None:
        warn_external(
            "Passing win_size with gaussian_weights=True is deprecated "
            "and will raise an error in a future version. The window "
            "size is determined by sigma; use sigma to control the "
            "effective window size.",
            category=FutureWarning,
        )
        win_size = None  # let _skimage2 derive it from sigma

    return ski2.metrics.structural_similarity(
        im1,
        im2,
        win_size=win_size,
        gradient=gradient,
        data_range=data_range,
        channel_axis=channel_axis,
        gaussian_weights=gaussian_weights,
        full=full,
        **kwargs,
    )


from skimage._doctest_adapters import adapt_doctests  # noqa: E402

adapt_doctests(globals())
