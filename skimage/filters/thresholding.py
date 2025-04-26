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
    hint="Use `skimage.segmentation.threshold_plot_all_global` instead",
)
@require("matplotlib", ">=3.3")
def try_all_threshold(image, figsize=(8, 5), verbose=True):
    from ..segmentation import _thresholding_global

    return _thresholding_global.threshold_plot_all_global(
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


@deprecate_func(
    deprecated_version="0.26",
    removed_version="2.0 (or later)",
    hint="Use `skimage.segmentation.threshold_labels_hsysteresis` instead",
)
def apply_hysteresis_threshold(image, low, high):
    from ..segmentation import _thresholding_local

    return _thresholding_local.threshold_labels_hysteresis(
        image=image, low=low, high=high
    )


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
