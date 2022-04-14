from .._shared import lazy

__getattr__, __dir__, __all__ = lazy.attach(
    'skimage.filters',
    submodules={''},
    submod_attrs={
        'lpi_filter': ['forward', 'inverse', 'wiener', 'LPIFilter2D'],
        '_gaussian': ['difference_of_gaussians', 'gaussian'],
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
        'ridges': ['meijering', 'sato', 'frangi', 'hessian',
                   'compute_hessian_eigenvalues'],
        '_median': ['median'],
        '_sparse': ['correlate_sparse'],
        '_unsharp_mask': ['unsharp_mask'],
        '_window': ['window'],
        '_fft_based': ['butterworth']
    }
)
