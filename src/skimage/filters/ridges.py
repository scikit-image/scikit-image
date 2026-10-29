"""

Ridge filters.

Ridge filters can be used to detect continuous edges, such as vessels,
neurites, wrinkles, rivers, and other tube-like structures. The present
class of ridge filters relies on the eigenvalues of the Hessian matrix of
image intensities to detect tube-like structures where the intensity changes
perpendicular but not along the structure.

"""

from _skimage2.filters.ridges import (
    frangi as frangi,
    hessian as hessian,
    meijering as meijering,
    sato as sato,
)  # noqa: F401

__all__ = [
    'frangi',
    'hessian',
    'meijering',
    'sato',
]

from skimage._doctest_adapters import adapt_doctests  # noqa: E402

adapt_doctests(globals())
