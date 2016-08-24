# -*- coding: utf-8 -*-
"""Image restoration module.

References
----------
.. [1] François Orieux, Jean-François Giovannelli, and Thomas
       Rodet, "Bayesian estimation of regularization and point
       spread function parameters for Wiener-Hunt deconvolution",
       J. Opt. Soc. Am. A 27, 1593-1607 (2010)

       http://www.opticsinfobase.org/josaa/abstract.cfm?URI=josaa-27-7-1593

.. [2] Richardson, William Hadley, "Bayesian-Based Iterative Method of
       Image Restoration". JOSA 62 (1): 55–59. doi:10.1364/JOSA.62.000055, 1972

.. [3] B. R. Hunt "A matrix theory proof of the discrete
       convolution theorem", IEEE Trans. on Audio and
       Electroacoustics, vol. au-19, no. 4, pp. 285-288, dec. 1971
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
