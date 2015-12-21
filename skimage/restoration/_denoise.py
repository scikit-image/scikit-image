# coding: utf-8
import numpy as np
from math import ceil
from .. import img_as_float
from ..restoration._denoise_cy import _denoise_bilateral, _denoise_tv_bregman
from .._shared.utils import skimage_deprecation, warn
import warnings
import pywt

try:
    import pywt
    pywt_available = True
except ImportError:
    pywt_available = False


def denoise_bilateral(image, win_size=None, sigma_color=None, sigma_spatial=1,
                      bins=10000, mode='constant', cval=0, multichannel=True,
                      sigma_range=None):
    """Denoise image using bilateral filter.

    This is an edge-preserving, denoising filter. It averages pixels based on
    their spatial closeness and radiometric similarity.

    Spatial closeness is measured by the Gaussian function of the Euclidean
    distance between two pixels and a certain standard deviation
    (`sigma_spatial`).

    Radiometric similarity is measured by the Gaussian function of the Euclidean
    distance between two color values and a certain standard deviation
    (`sigma_color`).

    Parameters
    ----------
    image : ndarray, shape (M, N[, 3])
        Input image, 2D grayscale or RGB.
    win_size : int
        Window size for filtering.
        If win_size is not specified, it is calculated as
        ``max(5, 2 * ceil(3 * sigma_spatial) + 1)``.
    sigma_color : float
        Standard deviation for grayvalue/color distance (radiometric
        similarity). A larger value results in averaging of pixels with larger
        radiometric differences. Note, that the image will be converted using
        the `img_as_float` function and thus the standard deviation is in
        respect to the range ``[0, 1]``. If the value is ``None`` the standard
        deviation of the ``image`` will be used.
    sigma_spatial : float
        Standard deviation for range distance. A larger value results in
        averaging of pixels with larger spatial differences.
    bins : int
        Number of discrete values for Gaussian weights of color filtering.
        A larger value results in improved accuracy.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
        How to handle values outside the image borders. See
        `numpy.pad` for detail.
    cval : string
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    multichannel : bool
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.

    Returns
    -------
    denoised : ndarray
        Denoised image.

    References
    ----------
    .. [1] http://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf

    Examples
    --------
    >>> from skimage import data, img_as_float
    >>> astro = img_as_float(data.astronaut())
    >>> astro = astro[220:300, 220:320]
    >>> noisy = astro + 0.6 * astro.std() * np.random.random(astro.shape)
    >>> noisy = np.clip(noisy, 0, 1)
    >>> denoised = denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15)
    """
    if multichannel:
        if image.ndim != 3:
            if image.ndim == 2:
                raise ValueError("Use ``multichannel=False`` for 2D grayscale "
                                 "images. The last axis of the input image "
                                 "must be multiple color channels not another "
                                 "spatial dimension.")
            else:
                raise ValueError("Bilateral filter is only implemented for "
                                 "2D grayscale images (image.ndim == 2) and "
                                 "2D multichannel (image.ndim == 3) images, "
                                 "but the input image has {0} dimensions. "
                                 "".format(image.ndim))
        elif image.shape[2] not in (3, 4):
            if image.shape[2] > 4:
                warnings.warn("The last axis of the input image is interpreted "
                              "as channels. Input image with shape {0} has {1} "
                              "channels in last axis. ``denoise_bilateral`` is "
                              "implemented for 2D grayscale and color images "
                              "only.".format(image.shape, image.shape[2]))
            else:
                msg = "Input image must be grayscale, RGB, or RGBA; " \
                      "but has shape {0}."
                warnings.warn(msg.format(image.shape))
    else:
        if image.ndim > 2:
            raise ValueError("Bilateral filter is not implemented for "
                             "grayscale images of 3 or more dimensions, "
                             "but input image has {0} dimension. Use "
                             "``multichannel=True`` for 2-D RGB "
                             "images.".format(image.shape))

    if sigma_range is not None:
        warn('`sigma_range` has been deprecated in favor of '
             '`sigma_color`. The `sigma_range` keyword argument '
             'will be removed in v0.14', skimage_deprecation)

        #If sigma_range is provided, assign it to sigma_color
        sigma_color = sigma_range

    if win_size is None:
        win_size = max(5, 2 * int(ceil(3 * sigma_spatial)) + 1)

    return _denoise_bilateral(image, win_size, sigma_color, sigma_spatial,
                              bins, mode, cval)


def denoise_tv_bregman(image, weight, max_iter=100, eps=1e-3, isotropic=True):
    """Perform total-variation denoising using split-Bregman optimization.

    Total-variation denoising (also know as total-variation regularization)
    tries to find an image with less total-variation under the constraint
    of being similar to the input image, which is controlled by the
    regularization parameter.

    Parameters
    ----------
    image : ndarray
        Input data to be denoised (converted using img_as_float`).
    weight : float
        Denoising weight. The smaller the `weight`, the more denoising (at
        the expense of less similarity to the `input`). The regularization
        parameter `lambda` is chosen as `2 * weight`.
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when::

            SUM((u(n) - u(n-1))**2) < eps

    max_iter : int, optional
        Maximal number of iterations used for the optimization.
    isotropic : boolean, optional
        Switch between isotropic and anisotropic TV denoising.

    Returns
    -------
    u : ndarray
        Denoised image.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Total_variation_denoising
    .. [2] Tom Goldstein and Stanley Osher, "The Split Bregman Method For L1
           Regularized Problems",
           ftp://ftp.math.ucla.edu/pub/camreport/cam08-29.pdf
    .. [3] Pascal Getreuer, "Rudin–Osher–Fatemi Total Variation Denoising
           using Split Bregman" in Image Processing On Line on 2012–05–19,
           http://www.ipol.im/pub/art/2012/g-tvd/article_lr.pdf
    .. [4] http://www.math.ucsb.edu/~cgarcia/UGProjects/BregmanAlgorithms_JacquelineBush.pdf

    """
    return _denoise_tv_bregman(image, weight, max_iter, eps, isotropic)


def _denoise_tv_chambolle_nd(im, weight=0.1, eps=2.e-4, n_iter_max=200):
    """Perform total-variation denoising on n-dimensional images.

    Parameters
    ----------
    im : ndarray
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : ndarray
        Denoised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.

    """

    ndim = im.ndim
    p = np.zeros((im.ndim, ) + im.shape, dtype=im.dtype)
    g = np.zeros_like(p)
    d = np.zeros_like(im)
    i = 0
    while i < n_iter_max:
        if i > 0:
            # d will be the (negative) divergence of p
            d = -p.sum(0)
            slices_d = [slice(None), ] * ndim
            slices_p = [slice(None), ] * (ndim + 1)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax+1] = slice(0, -1)
                slices_p[0] = ax
                d[slices_d] += p[slices_p]
                slices_d[ax] = slice(None)
                slices_p[ax+1] = slice(None)
            out = im + d
        else:
            out = im
        E = (d ** 2).sum()

        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        slices_g = [slice(None), ] * (ndim + 1)
        for ax in range(ndim):
            slices_g[ax+1] = slice(0, -1)
            slices_g[0] = ax
            g[slices_g] = np.diff(out, axis=ax)
            slices_g[ax+1] = slice(None)

        norm = np.sqrt((g ** 2).sum(axis=0))[np.newaxis, ...]
        E += weight * norm.sum()
        tau = 1. / (2.*ndim)
        norm *= tau / weight
        norm += 1.
        p -= tau * g
        p /= norm
        E /= float(im.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if np.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out


def denoise_tv_chambolle(im, weight=0.1, eps=2.e-4, n_iter_max=200,
                         multichannel=False):
    """Perform total-variation denoising on n-dimensional images.

    Parameters
    ----------
    im : ndarray of ints, uints or floats
        Input data to be denoised. `im` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that
        determines the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.
    multichannel : bool, optional
        Apply total-variation denoising separately for each channel. This
        option should be true for color images, otherwise the denoising is
        also applied in the channels dimension.

    Returns
    -------
    out : ndarray
        Denoised image.

    Notes
    -----
    Make sure to set the multichannel parameter appropriately for color images.

    The principle of total variation denoising is explained in
    http://en.wikipedia.org/wiki/Total_variation_denoising

    The principle of total variation denoising is to minimize the
    total variation of the image, which can be roughly described as
    the integral of the norm of the image gradient. Total variation
    denoising tends to produce "cartoon-like" images, that is,
    piecewise-constant images.

    This code is an implementation of the algorithm of Rudin, Fatemi and Osher
    that was proposed by Chambolle in [1]_.

    References
    ----------
    .. [1] A. Chambolle, An algorithm for total variation minimization and
           applications, Journal of Mathematical Imaging and Vision,
           Springer, 2004, 20, 89-97.

    Examples
    --------
    2D example on astronaut image:

    >>> from skimage import color, data
    >>> img = color.rgb2gray(data.astronaut())[:50, :50]
    >>> img += 0.5 * img.std() * np.random.randn(*img.shape)
    >>> denoised_img = denoise_tv_chambolle(img, weight=60)

    3D example on synthetic data:

    >>> x, y, z = np.ogrid[0:20, 0:20, 0:20]
    >>> mask = (x - 22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2
    >>> mask = mask.astype(np.float)
    >>> mask += 0.2*np.random.randn(*mask.shape)
    >>> res = denoise_tv_chambolle(mask, weight=100)

    """

    im_type = im.dtype
    if not im_type.kind == 'f':
        im = img_as_float(im)

    if multichannel:
        out = np.zeros_like(im)
        for c in range(im.shape[-1]):
            out[..., c] = _denoise_tv_chambolle_nd(im[..., c], weight, eps,
                                                   n_iter_max)
    else:
        out = _denoise_tv_chambolle_nd(im, weight, eps, n_iter_max)
    return out


def _wavelet_threshold(img, wavelet, threshold=None, sigma=None, mode='soft'):
    """Performs wavelet denoising.

    Parameters
    ----------
    img : ndarray (2d or 3d) of ints, uints or floats
        Input data to be denoised. `img` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    wavelet : string
        The type of wavelet to perform. Can be any of the options
        pywt.wavelist outputs. For example, this may be any of ``{db1, db2,
        db3, db4, haar}``.
    sigma : float, optional
        The standard deviation of the noise. The noise is estimated when sigma
        is None (the default) by the method in [2]_.
    threshold : float, optional
        The thresholding value. All wavelet coefficients less than this value
        are set to 0. The default value (None) uses the SureShrink method found
        in [1]_ to remove noise.
    mode : {'soft', 'hard'}, optional
        An optional argument to choose the type of denoising performed. It
        noted that choosing soft thresholding given additive noise finds the
        best approximation of the original image.

    Returns
    -------
    out : ndarray
        Denoised image.

    References
    ----------
    .. [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
           thresholding for image denoising and compression." Image Processing,
           IEEE Transactions on 9.9 (2000): 1532-1546.
           DOI: 10.1109/83.862633
    .. [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
           by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
           DOI: 10.1093/biomet/81.3.425

    """
    coeffs = pywt.wavedecn(img, wavelet=wavelet)
    detail_coeffs = coeffs[-1]['d' * img.ndim]

    if sigma is None:
        # Estimates via the noise via method in [2]
        sigma = np.median(np.abs(detail_coeffs)) / 0.67448975019608171

    if threshold is None:
        # The BayesShrink threshold from [1]_ in docstring
        threshold = sigma**2 / np.sqrt(max(img.var() - sigma**2, 0))

    denoised_detail = [{key: pywt.threshold(level[key], value=threshold,
                       mode=mode) for key in level} for level in coeffs[1:]]
    denoised_root = pywt.threshold(coeffs[0], value=threshold, mode=mode)
    denoised_coeffs = [denoised_root] + [d for d in denoised_detail]
    return pywt.waverecn(denoised_coeffs, wavelet)


def denoise_wavelet(img, sigma=None, wavelet='db1', mode='soft',
                    multichannel=False):
    """Performs wavelet denoising on an image.

    Parameters
    ----------
    img : ndarray ([M[, N[, ...P]][, C]) of ints, uints or floats
        Input data to be denoised. `img` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    sigma : float, optional
        The noise standard deviation used when computing the threshold
        adaptively as described in [1]_. When None (default), the noise
        standard deviation is estimated via the method in [2]_.
    wavelet : string, optional
        The type of wavelet to perform and can be any of the options
        ``pywt.wavelist`` outputs. The default is `'db1'`. For example,
        ``wavelet`` can be any of ``{'db2', 'haar', 'sym9'}`` and many more.
    mode : {'soft', 'hard'}, optional
        An optional argument to choose the type of denoising performed. It
        noted that choosing soft thresholding given additive noise finds the
        best approximation of the original image.
    multichannel : bool, optional
        Apply wavelet denoising separately for each channel (where channels
        correspond to the final axis of the array).

    Returns
    -------
    out : ndarray
        Denoised image.

    Notes
    -----
    The wavelet domain is a sparse representation of the image, and can be
    thought of similarly to the frequency domain of the Fourier transform.
    Sparse representations have most values zero or near-zero and truly random
    noise is (usually) represented by many small values in the wavelet domain.
    Setting all values below some threshold to 0 reduces the noise in the
    image, but larger thresholds also decrease the detail present in the image.

    If the input is 3D, this function performs wavelet denoising on each color
    plane separately. The output image is clipped between either [-1, 1] and
    [0, 1] depending on the input image range.

    References
    ----------
    .. [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
           thresholding for image denoising and compression." Image Processing,
           IEEE Transactions on 9.9 (2000): 1532-1546.
           DOI: 10.1109/83.862633
    .. [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
           by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
           DOI: 10.1093/biomet/81.3.425

    Examples
    --------
    >>> from skimage import color, data
    >>> img = img_as_float(data.astronaut())
    >>> img = color.rgb2gray(img)
    >>> img += 0.1 * np.random.randn(*img.shape)
    >>> img = np.clip(img, 0, 1)
    >>> denoised_img = denoise_wavelet(img, sigma=0.1)

    """

    img = img_as_float(img)

    if multichannel:
        out = np.empty_like(img)
        for c in range(img.shape[-1]):
            out[..., c] = _wavelet_threshold(img[..., c], wavelet=wavelet,
                                             mode=mode, sigma=sigma)
    else:
        out = _wavelet_threshold(img, wavelet=wavelet, mode=mode,
                                 sigma=sigma)

    clip_range = (-1, 1) if img.min() < 0 else (0, 1)
    return np.clip(out, *clip_range)


def _sigma_est_dwt(detail_coeffs, distribution='Gaussian'):
    """
    Calculation of the robust median estimator of the noise standard
    deviation [1]_.

    Parameters
    ----------
    detail_coeffs : ndarray
        The detail coefficients corresponding to the discrete wavelet
        transform of an image.
    distribution : str
        The underlying noise distribution.

    Returns
    -------
    sigma : float
        estimated noise standard deviation.  See [1]_

    References
    ----------
    .. [1] Donoho, David L., and Jain M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
    """
    detail_coeffs = np.asarray(detail_coeffs)
    if distribution.lower() == 'gaussian':
        # 75th quantile of the underlying, symmetric noise distribution:
        # denom = scipy.stats.norm.ppf(0.75)
        denom = 0.67448975019608171
        sigma = np.median(np.abs(detail_coeffs)) / denom
    else:
        raise ValueError("Only Gaussian noise estimation is currently "
                         "supported")
    return sigma


def estimate_sigma(im, multichannel, average_sigmas=True):
    """
    Robust wavelet-based estimator of the noise standard deviation [1]_.

    Parameters
    ----------
    im : ndarray
        Image for which to estimate the noise standard deviation.
    multichannel : bool
       Estimate sigma separately for each channel.
    average_sigmas : bool, optional
        If true, average the channel estimates of `sigma`.  Otherwise return
        a list of sigmas corresponding to each channel.

    Returns
    -------
    sigma : float or list
        Estimated noise standard deviation(s).  If `multichannel` is True and
        `average_sigmas` is False, a separate noise estimate for each channel
        is returned.  Otherwise, the average of the individual channel
        estimates is returned.

    References
    ----------
    .. [1] Donoho, David L., and Jain M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.

    Examples
    --------
    >>> import skimage.data
    >>> from skimage import img_as_float
    >>> img = img_as_float(skimage.data.camera())
    >>> sigma = 0.1
    >>> img = img + sigma * np.random.standard_normal(img.shape)
    >>> sigma_hat = estimate_sigma(img, multichannel=False)
    """
    if not pywt_available:
        raise ValueError("estimate_sigma requires PyWavelets to be installed.")

    im = np.asarray(im)

    if multichannel:
        kwargs = dict(multichannel=False, distribution='Gaussian')
        nchannels = im.shape[-1]
        sigmas = [estimate_sigma(
            im[..., c], **kwargs) for c in range(nchannels)]
        if average_sigmas:
            sigmas = np.mean(sigmas)
        return sigmas

    coeffs = pywt.dwtn(im, wavelet='db2')
    detail_coeffs = coeffs['d' * im.ndim]
    return _sigma_est_dwt(detail_coeffs, distribution='Gaussian')
