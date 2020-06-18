import sys
from ._array_like import _SupportsArray, ArrayLike
from ._shape import _Shape, _ShapeLike
from ._dtype_like import DtypeLike
from ._image_types import (ImageArray, LabelsArray, Image2dArray,
                           ImagendArray, MaskArray)

from typing import Any
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
