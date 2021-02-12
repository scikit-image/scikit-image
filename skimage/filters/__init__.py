from .lpi_filter import inverse, wiener, LPIFilter2D
from ._gaussian import (gaussian, _guess_spatial_dimensions,
                        difference_of_gaussians)
from .edges import (sobel, sobel_h, sobel_v,
                    scharr, scharr_h, scharr_v,
                    prewitt, prewitt_h, prewitt_v,
                    roberts, roberts_pos_diag, roberts_neg_diag,
                    laplace,
                    farid, farid_h, farid_v)
from ._rank_order import rank_order
from ._gabor import gabor_kernel, gabor
from .thresholding import (threshold_local, threshold_otsu, threshold_yen,
                           threshold_isodata, threshold_li, threshold_minimum,
                           threshold_mean, threshold_triangle,
                           threshold_niblack, threshold_sauvola,
                           threshold_multiotsu, try_all_threshold,
                           apply_hysteresis_threshold)
from .ridges import (meijering, sato, frangi, hessian)
from . import rank
from ._median import median
from ._sparse import correlate_sparse
from ._unsharp_mask import unsharp_mask
from ._window import window


__all__ = ['inverse',
           'correlate_sparse',
           'wiener',
           'LPIFilter2D',
           'gaussian',
           'difference_of_gaussians',
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
           'farid',
           'farid_h',
           'farid_v',
           'rank_order',
           'gabor_kernel',
           'gabor',
           'try_all_threshold',
           'meijering',
           'sato',
           'frangi',
           'hessian',
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
           'threshold_multiotsu',
           'apply_hysteresis_threshold',
           'rank',
           'unsharp_mask',
           'window']

from ..util import lazy
print('module:', __name__)
__getattr__, __dir__, __all__ = lazy.install_lazy(
    __name__,
    submodules={'rank'},
    submod_attrs={
        'lpi_filter': ['inverse', 'wiener', 'LPFilter2D'],
        '_gaussian': ['gaussian', '_guess_spatial_dimensions',
                      'difference_of_gaussians'],
        'edges': ['sobel', 'sobel_h', 'sobel_v',
                  'scharr', 'scharr_h', 'scharr_v',
                  'prewitt', 'prewitt_h', 'prewitt_v',
                  'roberts', 'roberts_pos_diag', 'roberts_neg_diag',
                  'laplace',
                  'farid', 'farid_h', 'farid_v'],
        '_rank_order': ['rank_order'],
        '_gabor': ['gabor_kernel', 'gabor'],
        'thresholding': ['threshold_local', 'threshold_otsu', 'threshold_yen',
                         'threshold_isodata', 'threshold_li', 'threshold_minimum',
                         'threshold_mean', 'threshold_triangle',
                         'threshold_niblack', 'threshold_sauvola',
                         'threshold_multiotsu', 'try_all_threshold',
                         'apply_hysteresis_threshold'],
        'ridges': ['meijering', 'sato', 'frangi', 'hessian'],
        '_median': ['median'],
        '_sparse': ['correlate_sparse'],
        '_unsharp_mask': ['unsharp_mask'],
        '_window': ['window']
    }
)
