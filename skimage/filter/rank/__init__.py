from .generic import (autolevel, bottomhat, equalize, gradient, maximum, mean,
                      subtract_mean, median, minimum, modal, enhance_contrast,
                      pop, threshold, tophat, noise_filter, entropy, otsu)
from .percentile import (autolevel_percentile, gradient_percentile,
                         mean_percentile, subtract_mean_percentile,
                         enhance_contrast_percentile, percentile,
                         pop_percentile, threshold_percentile)
from .bilateral import mean_bilateral, pop_bilateral

from skimage._shared.utils import deprecated


percentile_autolevel = deprecated('autolevel_percentile')(autolevel_percentile)

percentile_gradient = deprecated('gradient_percentile')(gradient_percentile)

percentile_mean = deprecated('mean_percentile')(mean_percentile)
bilateral_mean = deprecated('mean_bilateral')(mean_bilateral)

meansubtraction = deprecated('subtract_mean')(subtract_mean)
percentile_mean_subtraction = deprecated('subtract_mean_percentile')\
                                        (subtract_mean_percentile)

morph_contr_enh = deprecated('enhance_contrast')(enhance_contrast)
percentile_morph_contr_enh = deprecated('enhance_contrast_percentile')\
                                       (enhance_contrast_percentile)

percentile_pop = deprecated('pop_percentile')(pop_percentile)
bilateral_pop = deprecated('pop_bilateral')(pop_bilateral)

percentile_threshold = deprecated('threshold_percentile')(threshold_percentile)


__all__ = ['autolevel',
           'autolevel_percentile',
           'bottomhat',
           'equalize',
           'gradient',
           'gradient_percentile',
           'maximum',
           'mean',
           'mean_percentile',
           'mean_bilateral',
           'subtract_mean',
           'subtract_mean_percentile',
           'median',
           'minimum',
           'modal',
           'enhance_contrast',
           'enhance_contrast_percentile',
           'pop',
           'pop_percentile',
           'pop_bilateral',
           'threshold',
           'threshold_percentile',
           'tophat',
           'noise_filter',
           'entropy',
           'otsu'
           'percentile',
           # Deprecated
           'percentile_autolevel',
           'percentile_gradient',
           'percentile_mean',
           'percentile_mean_subtraction',
           'percentile_morph_contr_enh',
           'percentile_pop',
           'percentile_threshold',
           'bilateral_mean',
           'bilateral_pop']
