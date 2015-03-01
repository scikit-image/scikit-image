from .._shared.utils import skimage_deprecation
from warnings import warn

global _import_warned

warn(skimage_deprecation('The `skimage.filter` module has been renamed '
                         'to `skimage.filters`.  This placeholder module '
                         'will be removed in v0.13.'))
_import_warned = True

del warn
del skimage_deprecation

from ..filters.lpi_filter import inverse, wiener, LPIFilter2D
from ..filters._gaussian import gaussian_filter
from ..filters.edges import (sobel, hsobel, vsobel, sobel_h, sobel_v,
                    scharr, hscharr, vscharr, scharr_h, scharr_v,
                    prewitt, hprewitt, vprewitt, prewitt_h, prewitt_v,
                    roberts, roberts_positive_diagonal,
                    roberts_negative_diagonal, roberts_pos_diag,
                    roberts_neg_diag)
from ..filters._rank_order import rank_order
from ..filters._gabor import gabor_kernel, gabor_filter
from ..filters.thresholding import (threshold_adaptive, threshold_otsu, threshold_yen,
                           threshold_isodata)
from ..filters import rank
from ..filters.rank import median

from skimage._shared.utils import deprecated
from skimage import restoration
denoise_bilateral = deprecated('skimage.restoration.denoise_bilateral')\
                        (restoration.denoise_bilateral)
denoise_tv_bregman = deprecated('skimage.restoration.denoise_tv_bregman')\
                        (restoration.denoise_tv_bregman)
denoise_tv_chambolle = deprecated('skimage.restoration.denoise_tv_chambolle')\
                        (restoration.denoise_tv_chambolle)

# Backward compatibility v<0.11
@deprecated('skimage.feature.canny')
def canny(*args, **kwargs):
    # Hack to avoid circular import
    from skimage.feature._canny import canny as canny_
    return canny_(*args, **kwargs)


__all__ = ['inverse',
           'wiener',
           'LPIFilter2D',
           'gaussian_filter',
           'median',
           'canny',
           'sobel',
           'hsobel',
           'vsobel',
           'sobel_h',
           'sobel_v',
           'scharr',
           'hscharr',
           'vscharr',
           'scharr_h',
           'scharr_v',
           'prewitt',
           'hprewitt',
           'vprewitt',
           'prewitt_h',
           'prewitt_v',
           'roberts',
           'roberts_positive_diagonal',
           'roberts_negative_diagonal',
           'roberts_pos_diag',
           'roberts_neg_diag',
           'denoise_tv_chambolle',
           'denoise_bilateral',
           'denoise_tv_bregman',
           'rank_order',
           'gabor_kernel',
           'gabor_filter',
           'threshold_adaptive',
           'threshold_otsu',
           'threshold_yen',
           'threshold_isodata',
           'rank']
