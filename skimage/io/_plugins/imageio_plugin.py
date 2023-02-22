__all__ = ['imread', 'imsave']

from functools import wraps
import numpy as np

from imageio.v3 import imread as imageio_imread, imwrite as imsave


@wraps(imageio_imread)
def imread(*args, **kwargs):
    return np.asarray(imageio_imread(*args, **kwargs))
