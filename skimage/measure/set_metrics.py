from __future__ import print_function, division

import numpy as np
from ..morphology import binary_erosion, disk, ball

from ._set_metrics import hausdorff_distance_onesided


def coordinates(shape):
    for i, s in enumerate(shape):
        c = np.ones(shape, dtype=np.int)
        c *= np.arange(s).reshape(tuple([s if i == j else 1
                                         for j in range(len(shape))]))
        yield c


def find_coordinate_list(image):
    if image.dtype != np.bool:
        raise ValueError('image must have dtype = \'bool\'')
    return np.vstack(c[image] for c in coordinates(image.shape)).T


def binary_find_boundaries(image):
    if image.dtype != np.bool:
        raise ValueError('image must have dtype = \'bool\'')
    if image.ndim == 2:
        selem = disk(1)
    elif image.ndim == 3:
        selem = ball(1)
    else:
        raise ValueError('image must be 2D or 3D')
    eroded = binary_erosion(image, selem)
    return (image & (~eroded))


def hausdorff_distance(a, b):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('Both input arrays must be two-dimensional')
    if a.shape[1] != b.shape[1]:
        raise ValueError('Second dimension of the arrays must be equal')
    a = np.require(a, np.float64, ['C'])
    b = np.require(b, np.float64, ['C'])
    return max(hausdorff_distance_onesided(a, b),
               hausdorff_distance_onesided(b, a))


def hausdorff_distance_region(a, b):
    if a.dtype != np.bool or b.dtype != np.bool:
        raise ValueError('Arrays must have dtype = \'bool\'')
    if a.shape != b.shape:
        raise ValueError('Array shapes must be identical')

    a_points = find_coordinate_list(binary_find_boundaries(a))
    b_points = find_coordinate_list(binary_find_boundaries(b))
    return hausdorff_distance(a_points, b_points)
