"""

Implementation of the masked normalized cross-correlation.

Based on the following publication:
D. Padfield. Masked object registration in the Fourier domain.
IEEE Transactions on Image Processing (2012)

and the author's original MATLAB implementation, available on this website:
http://www.dirkpadfield.com/

"""

from _skimage2.registration._masked_phase_cross_correlation import (
    cross_correlate_masked as cross_correlate_masked,
)  # noqa: F401

__all__ = ['cross_correlate_masked']

from _skimage2.registration._masked_phase_cross_correlation import (  # noqa: F401
    _masked_phase_cross_correlation,
)

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
