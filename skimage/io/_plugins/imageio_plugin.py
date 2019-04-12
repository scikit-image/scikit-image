__all__ = ['imread', 'imsave']

from functools import wraps
import numpy as np
from imageio import imread as imageio_imread, imsave


def array_return(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return np.asarray(f(*args, **kwargs))
    return wrapper


imread = array_return(imageio_imread)
