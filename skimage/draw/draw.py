'''
Methods to draw on arrays.

'''

import numpy as np
from ._draw import bresenham, _fill_polygon


def fill_polygon(image, coords, color=1):
    '''
    Fill polygon in image with scanline-algorithm.

    Parameters
    ----------
    image : ndarray
        image in which to store filled polygon
    coords : ndarray
        Nx2 array containing x, y coordinates of polygon
    color : integer, optional
        face color of polygon, default: 1

    Reference: http://www.cs.ucdavis.edu/~ma/ECS175_S00/Notes/0411_b.pdf
    '''

    if image.ndim != 2:
        raise TypeError('The input image must be a two dimensional array.')
    if not np.issubdtype(image.dtype, 'uint8'):
        raise TypeError('The input image dtype must be \'uint8\'.')
    if coords.ndim != 2:
        raise TypeError('The coordinates must be a two dimensional array.')
    if np.any(coords[0,:] != coords[-1,:]):
        raise ValueError('Last coordinate must equal first.')
    _fill_polygon(image, coords.astype('double'), color)
