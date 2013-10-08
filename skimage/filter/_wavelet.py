import numpy as np
from scipy.misc import imresize, bytescale
try:
    import pywt
except ImportError:
    raise ImportError("pwavelets must be installed to use wavelet filter \
                      functions")

__all__ = ['wavelet_filter', 'wavelet_coefficient_array', 'wavelet_list']

_thresh_func = {"soft": pywt.thresholding.soft,
                "hard": pywt.thresholding.hard}


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

    If wavelet coefficients are needed for analysis, wavedec2 should be
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
        cH_, cV_, cD_ = imresize(cH, sh), imresize(cV, sh), imresize(cD, sh)
        temp = np.empty((2 * sh[0], 2 * sh[1]))
        temp[:sh[0], :sh[1]] = cA
        temp[sh[0]:, :sh[1]] = bytescale(cH_, low=cH.min(), high=cH.max())
        temp[:sh[0], sh[1]:] = bytescale(cV_, low=cV.min(), high=cV.max())
        temp[sh[0]:, sh[1]:] = bytescale(cD_, low=cD.min(), high=cD.max())
        cA = temp
    cA = bytescale(imresize(cA, (m, n)), low=cA.min(), high=cA.max())
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

    if not coeffs:
        coeffs = pywt.wavedec2(image, wavelet, level=level, mode=mode)

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
