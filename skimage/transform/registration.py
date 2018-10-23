import numpy as np
from scipy.optimize import basinhopping, minimize
from scipy import ndimage as ndi

from .pyramids import pyramid_gaussian
from ..measure import compare_mse

__all__ = ['register_affine']


def _p_to_matrix(p):
    """
    Converts a transformation in form of (r, tc, tr) into a 3x3 transformation matrix

    Parameters
    ----------
    param : array
        Input array giving the translation and rotation of an image

    Returns
    -------
    matrix : (3, 3) array
        A transformation matrix used to obtain a new image
    """

    out = np.empty((3, 3))

    out[0][0] = p[0]
    out[0][1] = p[1]
    out[0][2] = p[2]
    out[1][0] = p[3]
    out[1][1] = p[4]
    out[1][2] = p[5]
    out[2][0] = 0
    out[2][1] = 0
    out[2][2] = 1

    return out


def _matrix_to_p(matrix):
    """
    Converts a 3x3 transformation matrix into a transformation in form of (r, tc, tr)

    Parameters
    ----------
    matrix : (3, 3) array
        A transformation matrix used to obtain a new image

    Returns
    -------
    param : array
        Input array giving the translation and rotation of an image

    """

    return (matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][0], matrix[1][1], matrix[1][2])


def register_affine(reference, target, *, cost=compare_mse, nlevels=7,
                    iter_callback=lambda img, matrix: None):

    pyramid_ref = pyramid_gaussian(reference, max_layer=nlevels - 1)
    pyramid_tgt = pyramid_gaussian(target, max_layer=nlevels - 1)
    image_pairs = reversed(list(zip(pyramid_ref, pyramid_tgt)))
    p = np.zeros(6)
    p[0] = 1
    p[4] = 1

    for (ref, tgt) in image_pairs:
        def _cost(param):
            transformation = _p_to_matrix(param)
            transformed = ndi.affine_transform(tgt, transformation, order=1)
            return cost(ref, transformed)

        res = minimize(_cost, p, method='Powell')
        p = res.x
        iter_callback(tgt, _p_to_matrix(p))

    matrix = _p_to_matrix(p)

    return matrix
