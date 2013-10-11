import numpy as np
from functools import reduce
from scipy.interpolate import RectBivariateSpline
try:
    import pywt
except ImportError:
    raise ImportError("pwavelets must be installed to use wavelet filter \
                      functions")

__all__ = ['wavelet_filter', 'wavelet_coefficient_array', 'wavelet_list',
           'bayes_shrink', 'visu_shrink']

_thresh_func = {"soft": pywt.thresholding.soft,
                "hard": pywt.thresholding.hard}


def ca_resize(image, size):
    """
    Resizes coefficient arrays using bivariate spline approximation.
    """
    m, n = image.shape
    X = np.linspace(0, m - 1, size[0])
    Y = np.linspace(0, n - 1, size[1])
    kx, ky = min([m - 1, 3]), min([n - 1, 3])
    interp = RectBivariateSpline(
        np.arange(m), np.arange(n), image, kx=kx, ky=ky)
    resized = interp(X, Y)
    return resized


def wavelet_thresholding_based_filter(func):
    """
    Decorator for wavelet filtering methods based on
    coefficient thresholding.

    Calls decorated function to get threshold levels, then passes
    them (along with pre-calculated wavelet coefficients) to wavelet_filter().

    Decorated functions need only implement the way that they
    calculate thresholds.
    """
    def perform_filter(image, *args, **kw):
        thresholds, coeffs = func(image, *args, **kw)
        if hasattr(thresholds, "__iter__"):
            thresholds = thresholds[::-1]
        return wavelet_filter(image, thresholds, coeffs=coeffs, *args, **kw)

    return perform_filter


def _universal_threshold(coeffs):
    """
    Calculation of Donoho and Johnstone's `universal` wavelet
    coefficient threshold[1]. Used in various denoising methods such as
    BayesShrink and VisuShrink.

    [1] Donoho, David L., and Jain M. Johnstone. "Ideal spatial adaptation by
        wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
    """
    return np.median(abs(coeffs[-1][2])) / 0.6745


@wavelet_thresholding_based_filter
def visu_shrink(image, wavelet="haar", thresh_type="soft", level=1,
                mode='sym', coeffs=None):
    """
    VisuShrink[1] image denoising filter.

    Parameters
    ----------

    image : array-like
        input grayscale image to filter.

    Optional
    --------
    wavelet : string
              name of the wavelet filter to use. defaults to "haar"
    level : int
            the number of wavelet decomposition levels to perform
            default:  1
    mode: string
          signal extension mode for the wavelet decompostion. default: `sym`
    coeffs: list of wavelet coefficients (same format as wavedec2 output)
        if wavelet coefficients are provided by the user, they will be
        used directly and not computed.

    Returns
    -------

    filtered_image : ndarray
        the filtered array

    References
    ----------
    [1] Donoho, David L., and Jain M. Johnstone. "Ideal spatial adaptation by
        wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
    """
    if not coeffs:
        coeffs = pywt.wavedec2(image, wavelet, level=level, mode=mode)

    s_hat = _universal_threshold(coeffs)
    threshold = s_hat * np.sqrt(2 * np.log10(image.size))

    return threshold, coeffs


@wavelet_thresholding_based_filter
def bayes_shrink(image, wavelet="haar", thresh_type="soft", level=1,
                 mode='sym', coeffs=None):
    """
    BayesShrink image denoising filter.

    BayesShrink is a subband-adaptive data-driven method for image denoising
    via wavelet coefficient thresholding. The threshold is determined via
    Bayesian inference[1].

    Parameters
    ----------

    image : array-like
        input grayscale image to filter.

    Optional
    --------
    wavelet : string
              name of the wavelet filter to use. defaults to "haar"
    level : int
            the number of wavelet decomposition levels to perform
            default:  1
    mode: string
          signal extension mode for the wavelet decompostion. default: `sym`
    coeffs: list of wavelet coefficients (same format as wavedec2 output)
        if wavelet coefficients are provided by the user, they will be
        used directly and not computed.

    Returns
    -------

    filtered_image : ndarray
        the filtered array

    References
    ----------
    [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
        thresholding for image denoising and compression." Image Processing,
        IEEE Transactions on 9.9 (2000): 1532-1546.
    """
    if not coeffs:
        coeffs = pywt.wavedec2(image, wavelet, level=level, mode=mode)

    s_hat = _universal_threshold(coeffs)
    thresholds = []
    for level_bands in coeffs[1:]:
        thresholds.append([s_hat ** 2 / np.sqrt(max([1e-16,
                           C.std() ** 2 - s_hat ** 2])) for C in level_bands])

    return thresholds, coeffs


def wavelet_list():
    """
    Returns list of all wavelet functions currently implemented in pywt

    Parameters
    ----------
    (None)

    Returns
    -------
    wavelet_list: list
        List of names of all wavelet functions currently implemented by pywt
    """
    wavelets = [pywt.wavelist(family) for family in pywt.families()]
    wavelet_list = reduce(lambda x, y: x + y, wavelets)
    return wavelet_list


def wavelet_coefficient_array(image, coeffs=None, wavelet="haar", level=1):
    """
    Computer the coefficients of the dicrete wavelet transform, and
    arrange them canonically into a single array for visualization.

    This function should be used for visualization purposes only; the returned
    wavelet coefficients should not be used for analysis.

    If wavelet coefficients are needed for analysis, pywt.wavedec2 should be
    called directly.

    Parameters
    ----------

    image: array-like
        input grayscale image

    Optional
    --------
    coeffs: list of wavelet coefficients (same format as wavedec2 output)
        if wavelet coefficients are provided by the user, they will be
        plotted directly and not computed. This can be used to visualize the
        effect of thresholding functions used in denoising applications.
    wavelet: string
        name of wavelet filter to use
    level: int
        number of decomposition levels

    Returns
    -------
    cA: array-like
        wavelet transform coefficient array
    """
    m, n = image.shape

    if not coeffs:
        coeffs = pywt.wavedec2(image, wavelet, level=level)

    cA = coeffs[0]
    sh = cA.shape
    for cH, cV, cD in coeffs[1:]:
        sh = cA.shape
        cH_, cV_, cD_ = ca_resize(cH, sh), ca_resize(cV, sh), ca_resize(cD, sh)
        temp = np.empty((2 * sh[0], 2 * sh[1]))
        temp[:sh[0], :sh[1]] = cA
        temp[sh[0]:, :sh[1]] = cH_
        temp[:sh[0], sh[1]:] = cV_
        temp[sh[0]:, sh[1]:] = cD_
        cA = temp
    cA = ca_resize(cA, (m, n))
    return cA


def wavelet_filter(image, thresholds, wavelet="haar", thresh_type="soft",
                   level=1, mode='sym', coeffs=None):
    """
    Filter based on the multi-level discrete wavelet transform

    Parameters
    ----------

    image : array-like
        input grayscale image to filter.
    thresholds : scalar or list of scalars
        Threshold levels for wavelet coefficient filtering. These will be
        applied to the detail coefficient subbands only.
      - If a scalar is given, this threshold will be applied uniformly to all
        detail subbands.
      - If a list of scalars is given that is the same size as the number
        of decomposition levels, then each threshold will be uniformly applied
        to its corresponding level
      - if a list of lists of scalars is given that is the same size as the
        number if of decomposition levels, and the size of each inner list is
        3, then there will be one threshold applied to each detail subband.

        Any other format for `threshold` will result in an exception.

        Level-dependent thresholds are specified in increasing order, i.e.
        the first element applies to the first level, second element to the
        second level, etc.

    Optional
    --------
    wavelet : string
              name of the wavelet filter to use. defaults to "haar"
    level : int
            the number of wavelet decomposition levels to perform
            default:  1
    mode: string
          signal extension mode for the wavelet decompostion. default: `sym`
    coeffs: list of wavelet coefficients (same format as wavedec2 output)
        if wavelet coefficients are provided by the user, they will be
        used directly and not computed.

    Returns
    -------

    filtered_image : ndarray
        the filtered array

    Notes
    -----

    pywavelets is used to perform the forward and inverse discrete
    wavelet transforms.

    Examples
    --------

    from scipy.misc import lena
    A = lena()
    # 1-level db4 transform, soft threshold all detail subbands at t=10.
    B = wavelet_filter(A, 10.,  wavelet="db4")
    # 3-level Haar transform, hard threshold each level with its own value
    t = [10., 8., 3.]
    B = wavelet_filter(A, t, thresh_type="hard", level=3)
    # 2-level sym14 transform, with each subband getting its own threshold
    t = [[10., 80., 30.], [80., 17., 19.]]
    B = wavelet_filter(A, t, wavelet="sym14", level=2)
    """
    if not coeffs:
        coeffs = pywt.wavedec2(image, wavelet, level=level, mode=mode)
    else:
        level = len(coeffs) - 1

    if isinstance(thresholds, float) or isinstance(thresholds, int):
        thresholds = [3 * [thresholds] for i in range(level)]

    elif hasattr(thresholds, '__iter__') and len(thresholds) == level:
        if isinstance(thresholds[0], float) or isinstance(thresholds[0], int):
            thresholds = [3 * [thresholds[i]] for i in range(level)]

        elif hasattr(thresholds, '__iter__') and len(thresholds[0]) == 3:
            pass
        else:
            raise Exception("Wavelet threshold values not set correctly.")
    else:
        raise Exception("Wavelet threshold values not set correctly.")

    new_coeffs = [coeffs[0]] + [_filt(bands, vals, thresh_type) for bands,
                                vals in zip(coeffs[1:], thresholds[::-1])]

    return pywt.waverec2(new_coeffs, wavelet, mode=mode)


def _filt(bands, vals, thresh_type):
    """
    Performs soft or hard wavelet coefficient thresholding on a single
    decomposition level
    """
    _thresh = _thresh_func[thresh_type]
    return [_thresh(band, t) for band, t in zip(bands, vals)]
