"""
canny.py - Canny Edge detector

Reference: Canny, J., A Computational Approach To Edge Detection, IEEE Trans.
    Pattern Analysis and Machine Intelligence, 8:679-714, 1986
"""

import warnings
import skimage2 as ski2
from ..util import PendingSkimage2Change


def canny(
    image,
    sigma=1.0,
    low_threshold=None,
    high_threshold=None,
    mask=None,
    use_quantiles=False,
    *,
    mode='constant',
    cval=0.0,
):
    """Edge filter an image using the Canny algorithm.

    Parameters
    ----------
    image : 2D array
        Grayscale input image to detect edges on; can be of any dtype.
    sigma : float, optional
        Standard deviation of the Gaussian filter.
    low_threshold : float, optional
        Lower bound for hysteresis thresholding (linking edges).
        If None, low_threshold is set to 10% of dtype's max.
    high_threshold : float, optional
        Upper bound for hysteresis thresholding (linking edges).
        If None, high_threshold is set to 20% of dtype's max.
    mask : array, dtype=bool, optional
        Mask to limit the application of Canny to a certain area.
    use_quantiles : bool, optional
        If ``True`` then treat low_threshold and high_threshold as
        quantiles of the edge magnitude image, rather than absolute
        edge magnitude values. If ``True`` then the thresholds must be
        in the range [0, 1].
    mode : str, {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        The ``mode`` parameter determines how the array borders are
        handled during Gaussian filtering, where ``cval`` is the value when
        mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if `mode` is 'constant'.

    Returns
    -------
    output : 2D array (image)
        The binary edge map.

    See also
    --------
    skimage.filters.sobel

    Notes
    -----
    The steps of the algorithm are as follows:

    * Smooth the image using a Gaussian with ``sigma`` width.

    * Apply the horizontal and vertical Sobel operators to get the gradients
      within the image. The edge strength is the norm of the gradient.

    * Thin potential edges to 1-pixel wide curves. First, find the normal
      to the edge at each point. This is done by looking at the
      signs and the relative magnitude of the X-Sobel and Y-Sobel
      to sort the points into 4 categories: horizontal, vertical,
      diagonal and antidiagonal. Then look in the normal and reverse
      directions to see if the values in either of those directions are
      greater than the point in question. Use interpolation to get a mix of
      points instead of picking the one that's the closest to the normal.

    * Perform a hysteresis thresholding: first label all points above the
      high threshold as edges. Then recursively label any point above the
      low threshold that is 8-connected to a labeled point as an edge.

    References
    ----------
    .. [1] Canny, J., A Computational Approach To Edge Detection, IEEE Trans.
           Pattern Analysis and Machine Intelligence, 8:679-714, 1986
           :DOI:`10.1109/TPAMI.1986.4767851`
    .. [2] William Green's Canny tutorial
           https://en.wikipedia.org/wiki/Canny_edge_detector

    Examples
    --------
    >>> from skimage import feature
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> # Generate noisy image of a square
    >>> im = np.zeros((256, 256))
    >>> im[64:-64, 64:-64] = 1
    >>> im += 0.2 * rng.random(im.shape)
    >>> # First trial with the Canny filter, with the default smoothing
    >>> edges1 = feature.canny(im)
    >>> # Increase the smoothing for better results
    >>> edges2 = feature.canny(im, sigma=3)

    """

    warnings.warn(
        "`skimage.feature.canny` is deprecated in favor of `skimage2.feature.canny`. "
        "The default of the optional parameter `mode` has been changed from `constant` to `nearest`. "
        "To keep the old (`skimage`, v1.x) behavior, use:\n"
        "\n"
        "    import skimage2 as ski2\n"
        "    ski2.feature.canny(\n"
        "        image,\n"
        "        mode='constant',\n"
        "        ...\n"
        "    )",
        stacklevel=2,
        category=PendingSkimage2Change,
    )

    return ski2.feature.canny(
        image,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        mask=mask,
        use_quantiles=use_quantiles,
        mode=mode,
        cval=cval,
    )
