# -*- coding: utf-8 -*-
"""Image restoration module.

"""

from .deconvolution import wiener, unsupervised_wiener, richardson_lucy
from .unwrap import unwrap_phase
from ._denoise import denoise_tv_chambolle, denoise_tv_bregman, \
                      denoise_bilateral, denoise_wavelet, estimate_sigma
from .non_local_means import denoise_nl_means
from .inpaint import inpaint_biharmonic
from .._shared.utils import copy_func, deprecated

nl_means_denoising = copy_func(denoise_nl_means, name='nl_means_denoising')
nl_means_denoising = deprecated('skimage.restoration.denoise_nl_means')(nl_means_denoising)


__all__ = ['wiener',
           'unsupervised_wiener',
           'richardson_lucy',
           'unwrap_phase',
           'denoise_tv_bregman',
           'denoise_tv_chambolle',
           'denoise_bilateral',
           'denoise_wavelet',
           'denoise_nl_means',
           'nl_means_denoising',
           'inpaint_biharmonic']

del copy_func, deprecated
