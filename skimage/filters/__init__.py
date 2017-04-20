from .lpi_filter import inverse, wiener, LPIFilter2D
from ._gaussian import gaussian
from .edges import (sobel, sobel_h, sobel_v,
                    scharr, scharr_h, scharr_v,
                    prewitt, prewitt_h, prewitt_v,
                    roberts, roberts_pos_diag, roberts_neg_diag,
                    laplace)
from ._rank_order import rank_order
from ._gabor import gabor_kernel, gabor, morlet_kernel
from ._frangi import frangi, hessian
from .thresholding import (threshold_local,
                           threshold_adaptive, threshold_otsu, threshold_yen,
                           threshold_isodata, threshold_li, threshold_minimum,
                           threshold_mean, threshold_triangle,
                           threshold_niblack, threshold_sauvola,
                           try_all_threshold)
from . import rank
from .rank import median
from .filter_bank import (multiresolution_filter_bank_morlet2d,
                          filter_bank_morlet2d,
                          filterbank_to_multiresolutionfilterbank)

from .._shared.utils import deprecated, copy_func


gaussian_filter = copy_func(gaussian, name='gaussian_filter')
gaussian_filter = deprecated('skimage.filters.gaussian')(gaussian_filter)
gabor_filter = copy_func(gabor, name='gabor_filter')
gabor_filter = deprecated('skimage.filters.gabor')(gabor_filter)

__all__ = ['inverse',
           'wiener',
           'LPIFilter2D',
           'gaussian',
           'median',
           'sobel',
           'sobel_h',
           'sobel_v',
           'scharr',
           'scharr_h',
           'scharr_v',
           'prewitt',
           'prewitt_h',
           'prewitt_v',
           'roberts',
           'roberts_pos_diag',
           'roberts_neg_diag',
           'laplace',
           'denoise_tv_chambolle',
           'denoise_bilateral',
           'denoise_tv_bregman',
           'rank_order',
           'gabor_kernel',
           'morlet_kernel',
           'gabor',
           'try_all_threshold',
           'frangi',
           'hessian',
           'threshold_adaptive',
           'threshold_otsu',
           'threshold_yen',
           'threshold_isodata',
           'threshold_li',
           'threshold_minimum',
           'threshold_mean',
           'threshold_niblack',
           'threshold_sauvola',
           'threshold_triangle',
           'rank',
           'multiresolution_filter_bank_morlet2d,'
           'filter_bank_morlet2d',
           'filterbank_to_multiresolutionfilterbank']
