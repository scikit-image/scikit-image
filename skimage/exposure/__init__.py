"""Exposure subpackage.

Image intensity and contrast adjustments via, e.g., simple rescaling of
intensities or histogram equalization [1]_.

.. [1] https://en.wikipedia.org/wiki/Histogram_equalization

"""


from .exposure import histogram, equalize_hist, \
                      rescale_intensity, cumulative_distribution, \
                      adjust_gamma, adjust_sigmoid, adjust_log, \
                      is_low_contrast

from ._adapthist import equalize_adapthist


__all__ = ['histogram',
           'equalize_hist',
           'equalize_adapthist',
           'rescale_intensity',
           'cumulative_distribution',
           'adjust_gamma',
           'adjust_sigmoid',
           'adjust_log',
           'is_low_contrast']
