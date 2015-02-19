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
                      denoise_bilateral
from .non_local_means import nl_means_denoising

__all__ = ['wiener',
           'unsupervised_wiener',
           'richardson_lucy',
           'unwrap_phase',
           'denoise_tv_bregman',
           'denoise_tv_chambolle',
           'denoise_bilateral',
           'nl_means_denoising']
