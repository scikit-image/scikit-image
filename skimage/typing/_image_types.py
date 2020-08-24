import numpy as np
from typing_extensions import Annotated

Image3D = Annotated[np.ndarray, 'image_3d',
                        'a 2D or 3D single or multichannel image']
Image2D = Annotated[np.ndarray, 'image_2d',
                        'a 2D single or multichannel image']
Image = Annotated[np.ndarray, 'image',
                            'an nD image']
Labels = Annotated[np.ndarray, 'labels',
                        'array of integers representing labels']
Mask = Annotated[np.ndarray, 'mask',
                        'array of bools to be used as mask']
