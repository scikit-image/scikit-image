import numpy as np
from scipy import ndimage
import warnings
try:
    import pywt
except ImportError:
    raise ImportError("pwavelets must be installed to use wavelet filter")

__all__ = ['wavelet_filter']

_thresh_func = {"soft": pywt.thresholding.soft,
                "hard": pywt.thresholding.hard}


def wavelet_filter(image, thresholds, wavelet="haar", thresh_type="soft",
                   level=1, mode='sym'):
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
    wavelet : string
              name of the wavelet filter to use. defaults to "haar"
    level : int
            the number of wavelet decomposition levels to perform
            default:  1
    mode: string
          signal extension mode for the wavelet decompostion. default: `sym`

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
    >>> from scipy.misc import lena
    >>> A = lena()
    >>> # 1-level db4 transform, soft threshold all detail subbands at t=10.
    >>> B = wavelet_filter(A, "db4", 10.)
    >>> # 3-level Haar transform, hard threshold each level with its own value
    >>> t = [10., 8., 3.]
    >>> B = wavelet_filter(A, "haar", t, thresh_type="hard", level=3)
    >>> # 2-level sym14 transform, with each subband getting its own threshold
    >>> t = [[10., 80., 30.], [80., 17., 19.]]
    >>> B = wavelet_filter(A, "sym14", t, level=2)
    """

    if isinstance(thresholds, float) or isinstance(thresholds, int):
        thresholds = [3 * [thresholds] for i in xrange(level)]

    elif isinstance(thresholds, list) and len(thresholds) == level:
        if isinstance(thresholds[0], float):
            thresholds = [3 * [thresholds[i]] for i in xrange(level)]

        elif isinstance(thresholds[0], list) and len(thresholds[0]) == 3:
            pass
        else:
            raise Exception("Wavelet threshold values not set correctly.")
    else:
        raise Exception("Wavelet threshold values not set correctly.")

    coeffs = pywt.wavedec2(image, wavelet, level=level, mode=mode)

    new_coeffs = [coeffs[0]] + [_filt(bands, vals, thresh_type) for bands,
                                vals in zip(coeffs[1:], thresholds)]

    return pywt.waverec2(new_coeffs, wavelet, mode=mode)


def _filt(bands, vals, thresh_type):
    """
    Performs soft or hard wavelet coefficient thresholding on a single
    decomposition level
    """
    _thresh = _thresh_func[thresh_type]
    return [_thresh(band, t) for band, t in zip(bands, vals)]

