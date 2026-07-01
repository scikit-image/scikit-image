from _skimage2.filters.thresholding import (
    try_all_threshold as try_all_threshold,
    threshold_otsu as threshold_otsu,
    threshold_yen as threshold_yen,
    threshold_isodata as threshold_isodata,
    threshold_li as threshold_li,
    threshold_local as threshold_local,
    threshold_minimum as threshold_minimum,
    threshold_mean as threshold_mean,
    threshold_niblack as threshold_niblack,
    threshold_sauvola as threshold_sauvola,
    threshold_triangle as threshold_triangle,
    apply_hysteresis_threshold as apply_hysteresis_threshold,
    threshold_multiotsu as threshold_multiotsu,
)  # noqa: F401

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

from _skimage2.filters.thresholding import _cross_entropy, _mean_std  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
