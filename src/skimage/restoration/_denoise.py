from _skimage2.restoration._denoise import (
    denoise_bilateral as denoise_bilateral,
    denoise_tv_bregman as denoise_tv_bregman,
    denoise_tv_chambolle as denoise_tv_chambolle,
    denoise_wavelet as denoise_wavelet,
    estimate_sigma as estimate_sigma,
)  # noqa: F401

__all__ = [
    'denoise_bilateral',
    'denoise_tv_bregman',
    'denoise_tv_chambolle',
    'denoise_wavelet',
    'estimate_sigma',
]

from _skimage2.restoration._denoise import _wavelet_threshold  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
