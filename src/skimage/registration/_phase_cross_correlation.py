"""

Port of Manuel Guizar's code from:
http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation

"""

from _skimage2.registration._phase_cross_correlation import (
    phase_cross_correlation as phase_cross_correlation,
)  # noqa: F401

__all__ = ['phase_cross_correlation']

from _skimage2.registration._phase_cross_correlation import _upsampled_dft  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
