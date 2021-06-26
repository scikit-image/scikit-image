import warnings

from .footprint import (
    square, rectangle, diamond, disk, cube, octahedron, ball, octagon, star
)

warnings.warn(
    "The `skimage.morphology.selem` module is deprecated and will be removed "
    "in scikit-image 1.0. The functions formerly in "
    "`skimage.morphology.selem` should be imported directly from "
    "`skimage.morphology` instead.",
    FutureWarning, stacklevel=2
)
