from _skimage2.transform._thin_plate_splines import (
    ThinPlateSplineTransform as ThinPlateSplineTransform,
)  # noqa: F401

__all__ = ['ThinPlateSplineTransform']

from skimage._docutils import bind_namespace
from skimage._pickle_compat import apply_pickle_exports

bind_namespace(globals())
apply_pickle_exports(globals())
