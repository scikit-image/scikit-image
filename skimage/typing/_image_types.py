import sys
from ._array_like import ArrayLike
import numpy as np
from typing import Type, Any

if sys.version_info >= (3, 8):
    from typing import Annotated
    HAVE_ANNOTATED = True
else:
    try:
        from typing_extensions import Annotated
    except ImportError:
        HAVE_ANNOTATED = False
    else:
        HAVE_ANNOTATED = True


ImageArray = Annotated[np.ndarray, 'image',
                        'a 2D or 3D single or multichannel image']
Image2dArray = Annotated[np.ndarray, 'image_2d']
ImagendArray = Annotated[np.ndarray, 'image_nd',
                            'nd image']
LabelsArray = Annotated[np.ndarray, 'labels',
                        'array of integers representing labels']
MaskArray = Annotated[np.ndarray, 'mask',
                        'array of bools to be used as mask']
