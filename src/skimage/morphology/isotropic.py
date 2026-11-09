"""

Binary morphological operations

"""

from _skimage2.morphology.isotropic import (
    isotropic_closing as isotropic_closing,
    isotropic_dilation as isotropic_dilation,
    isotropic_erosion as isotropic_erosion,
    isotropic_opening as isotropic_opening,
)  # noqa: F401

__all__ = [
    'isotropic_closing',
    'isotropic_dilation',
    'isotropic_erosion',
    'isotropic_opening',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
