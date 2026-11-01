from _skimage2.measure._regionprops_utils import (
    EULER_COEFS2D_4 as EULER_COEFS2D_4,
    EULER_COEFS2D_8 as EULER_COEFS2D_8,
    EULER_COEFS3D_26 as EULER_COEFS3D_26,
    STREL_4 as STREL_4,
    STREL_8 as STREL_8,
    euler_number as euler_number,
    perimeter as perimeter,
    perimeter_crofton as perimeter_crofton,
)  # noqa: F401

__all__ = [
    'EULER_COEFS2D_4',
    'EULER_COEFS2D_8',
    'EULER_COEFS3D_26',
    'STREL_4',
    'STREL_8',
    'euler_number',
    'perimeter',
    'perimeter_crofton',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
