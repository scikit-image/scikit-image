import sys
from ._array_like import _SupportsArray, ArrayLike
from ._shape import _Shape, _ShapeLike
from ._dtype_like import DtypeLike
from ._image_types import Image, Image2D, Image3D, Labels, Mask

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


__all__ = [
        '_SupportsArray'
        ]
