"""Image restoration module.

"""

from ._multimethods import (ball_kernel, calibrate_denoiser, cycle_spin,
                            denoise_bilateral, denoise_nl_means,
                            denoise_tv_bregman, denoise_tv_chambolle,
                            denoise_wavelet, ellipsoid_kernel, estimate_sigma,
                            inpaint_biharmonic, richardson_lucy, rolling_ball,
                            unsupervised_wiener, unwrap_phase, wiener)

__all__ = ['wiener',
           'unsupervised_wiener',
           'richardson_lucy',
           'unwrap_phase',
           'denoise_tv_bregman',
           'denoise_tv_chambolle',
           'denoise_bilateral',
           'denoise_wavelet',
           'denoise_nl_means',
           'estimate_sigma',
           'inpaint_biharmonic',
           'cycle_spin',
           'calibrate_denoiser',
           'rolling_ball',
           'ellipsoid_kernel',
           'ball_kernel',
           ]
