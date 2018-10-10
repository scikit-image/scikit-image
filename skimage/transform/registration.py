import numpy as np
from scipy.optimize import basinhopping, minimize
from scipy import ndimage as ndi

from .pyramids import pyramid_gaussian
from ..measure import compare_mse

__all__ = ['register_affine']


def _p_to_matrix(param):
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

    r, tc, tr = param
    out = np.empty((3, 3))

    out[0][0] = np.cos(r)
    out[0][1] = -1 * np.sin(r)
    out[0][2] = tc
    out[1][0] = np.sin(r)
    out[1][1] = np.cos(r)
    out[1][2] = tr
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

    return (np.arccos(matrix[0][0]), matrix[0][2], matrix[1][2])


def register_affine(reference, target, *, cost=compare_mse, nlevels=7,
                    method='Powell', iter_callback=lambda img, matrix: None):
    assert method in ['Powell', 'BH']

    pyramid_ref = pyramid_gaussian(reference, max_layer=nlevels - 1)
    pyramid_tgt = pyramid_gaussian(target, max_layer=nlevels - 1)
    image_pairs = reversed(list(zip(pyramid_ref, pyramid_tgt)))
    p = np.zeros(3)

    for n, (ref, tgt) in enumerate(image_pairs):
        def _cost(param):
            transformation = _p_to_matrix(param)
            transformed = ndi.affine_transform(tgt, transformation, order=1)
            return cost(ref, transformed)

        p[1:3] *= 2
        if method.upper() == 'BH':
            res = basinhopping(_cost, p)
            if n <= nlevels - 4:  # avoid basin-hopping in lower levels
                method = 'Powell'
        else:
            res = minimize(_cost, p, method='Powell')
        p = res.x
        iter_callback(tgt, _p_to_matrix(p))

    matrix = _p_to_matrix(p)

    return matrix
