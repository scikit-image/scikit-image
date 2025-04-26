import numpy as np
from scipy import ndimage as ndi

from .._shared.utils import deprecate_func
from .._shared.version_requirements import require


__all__ = [
    'try_all_threshold',
    'threshold_otsu',
    'threshold_yen',
    'threshold_isodata',
    'threshold_li',
    'threshold_local',
    'threshold_minimum',
    'threshold_mean',
    'threshold_niblack',
    'threshold_sauvola',
    'threshold_triangle',
    'apply_hysteresis_threshold',
    'threshold_multiotsu',
]


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation.threshold_try_global` instead",
)
@require("matplotlib", ">=3.3")
def try_all_threshold(image, figsize=(8, 5), verbose=True):
    from ..segmentation import _thresholding_global

    return _thresholding_global.threshold_try_global(
        image=image, figsize=figsize, verbose=verbose
    )


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation.threshold_local` instead",
)
def threshold_local(
    image, block_size=3, method='gaussian', offset=0, mode='reflect', param=None, cval=0
):
    from ..segmentation import _thresholding_local

    return _thresholding_local.threshold_local(
        image=image,
        block_size=block_size,
        method=method,
        offset=offset,
        mode=mode,
        param=param,
        cval=cval,
    )


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation.threshold_otsu` instead",
)
def threshold_otsu(image=None, nbins=256, *, hist=None):
    from ..segmentation import _thresholding_global

    return _thresholding_global.threshold_otsu(image=image, nbins=nbins, hist=hist)


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation.threshold_yen` instead",
)
def threshold_yen(image=None, nbins=256, *, hist=None):
    from ..segmentation import _thresholding_global

    return _thresholding_global.threshold_yen(image=image, nbins=nbins, hist=hist)


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation.threshold_isodata` instead",
)
def threshold_isodata(image=None, nbins=256, return_all=False, *, hist=None):
    from ..segmentation import _thresholding_global

    return _thresholding_global.threshold_isodata(
        image=image, nbins=nbins, return_all=return_all, hist=hist
    )


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation._thresholding_global._cross_entropy` instead",
)
def _cross_entropy(image, threshold, bins=None):
    from ..segmentation import _thresholding_global

    return _thresholding_global._cross_entropy(
        image=image, threshold=threshold, bins=bins
    )


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation.threshold_li` instead",
)
def threshold_li(image, *, tolerance=None, initial_guess=None, iter_callback=None):
    from ..segmentation import _thresholding_global

    return _thresholding_global.threshold_li(
        image=image,
        tolerance=tolerance,
        initial_guess=initial_guess,
        iter_callback=iter_callback,
    )


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation.threshold_minimum` instead",
)
def threshold_minimum(image=None, nbins=256, max_num_iter=10000, *, hist=None):
    from ..segmentation import _thresholding_global

    return _thresholding_global.threshold_minimum(
        image=image, nbins=nbins, max_num_iter=max_num_iter, hist=hist
    )


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation.threshold_mean` instead",
)
def threshold_mean(image):
    from ..segmentation import _thresholding_global

    return _thresholding_global.threshold_mean(image)


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation.threshold_triangle` instead",
)
def threshold_triangle(image, nbins=256):
    from ..segmentation import _thresholding_global

    return _thresholding_global.threshold_triangle(image=image, nbins=nbins)


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation.threshold_local_niblack` instead",
)
def threshold_niblack(image, window_size=15, k=0.2):
    from ..segmentation import _thresholding_local

    return _thresholding_local.threshold_local_niblack(
        image=image, window_size=window_size, k=k
    )


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation.threshold_local_sauvola` instead",
)
def threshold_sauvola(image, window_size=15, k=0.2, r=None):
    from ..segmentation import _thresholding_local

    return _thresholding_local.threshold_local_sauvola(
        image=image, window_size=window_size, k=k, r=r
    )


def apply_hysteresis_threshold(image, low, high):
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
    >>> apply_hysteresis_threshold(image, 1.5, 2.5).astype(int)
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


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation.threshold_multiotsu` instead",
)
def threshold_multiotsu(image=None, classes=3, nbins=256, *, hist=None):
    from ..segmentation import _thresholding_global

    return _thresholding_global.threshold_multiotsu(
        image=image, classes=classes, nbins=nbins, hist=hist
    )
