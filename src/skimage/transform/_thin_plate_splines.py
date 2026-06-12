from _skimage2.transform._thin_plate_splines import (
    ThinPlateSplineTransform as ThinPlateSplineTransform,
)  # noqa: F401

__all__ = ['ThinPlateSplineTransform']

from skimage._docutils import bind_namespace

bind_namespace(globals())
