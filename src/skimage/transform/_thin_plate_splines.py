from _skimage2.transform._thin_plate_splines import (
    ThinPlateSplineTransform as ThinPlateSplineTransform,
)  # noqa: F401

__all__ = ['ThinPlateSplineTransform']

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
