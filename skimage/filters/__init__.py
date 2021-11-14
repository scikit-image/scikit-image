from .._shared import lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules={'rank', '_api'},
    submod_attrs={
        'lpi_filter': ['LPIFilter2D'],
        '_multimethods': ['inverse', 'wiener', 'gaussian',
                          'difference_of_gaussians', 'sobel', 'sobel_h',
                          'sobel_v', 'scharr', 'scharr_h', 'scharr_v',
                          'prewitt', 'prewitt_h', 'prewitt_v', 'roberts',
                          'roberts_pos_diag', 'roberts_neg_diag', 'laplace',
                          'farid', 'farid_h', 'farid_v', 'rank_order',
                          'gabor_kernel', 'gabor', 'threshold_local',
                          'threshold_otsu', 'threshold_yen',
                          'threshold_isodata', 'threshold_li',
                          'threshold_minimum', 'threshold_mean',
                          'threshold_triangle',
                          'threshold_niblack', 'threshold_sauvola',
                          'threshold_multiotsu', 'try_all_threshold',
                          'apply_hysteresis_threshold', 'meijering', 'sato',
                          'frangi', 'hessian', 'median', 'correlate_sparse',
                          'unsharp_mask', 'window', 'butterworth',
                          # the following two shouldn't be here?
                          'compute_hessian_eigenvalues', 'forward']
    }
)
