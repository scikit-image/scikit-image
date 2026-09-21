from _skimage2.restoration.j_invariant import (
    calibrate_denoiser as calibrate_denoiser,
    denoise_invariant as denoise_invariant,
)  # noqa: F401

__all__ = [
    'calibrate_denoiser',
    'denoise_invariant',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
