import numpy as np
from scipy.optimize import basinhopping, minimize

from .pyramids import pyramid_gaussian
from ..measure import compare_mse
from ._warps import warp, SimilarityTransform

__all__ = ['register_affine', 'p_to_matrix', 'matrix_to_p']


def _cost_mse(param, im_true, im_test):
    """
    Finds the error between the overlapping area of two images after transformations are done to one of them

    Parameters
    ----------
    param : array
        Input array giving the translation and rotation the target image will undergo
    im_true : (M, N) ndarray
        Input image used for reference
    im_test : (M, N) ndarray
        Input image which is modified and then compared to im_true

    Returns
    -------
    err: int
        Error in the form of the mean of the squared difference between pixels
    """
    transformation = p_to_matrix(param)
    transformed = warp(im_test, transformation, order=3)
    return compare_mse(im_true, transformed)


def p_to_matrix(param):
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


def matrix_to_p(matrix):
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


def register_affine(reference, target, *, cost=_cost_mse, nlevels=7,
                    method='Powell', iter_callback=lambda img, p: None):
    assert method in ['Powell', 'BH']

    pyramid_ref = pyramid_gaussian(reference, max_layer=nlevels - 1)
    pyramid_tgt = pyramid_gaussian(target, max_layer=nlevels - 1)
    levels = range(nlevels, 0, -1)
    image_pairs = zip(pyramid_ref, pyramid_tgt)
    p = np.zeros(3)

    for n, (ref, tgt) in reversed(list(zip(levels, image_pairs))):
        p[1:3] *= 2
        if method.upper() == 'BH':
            res = basinhopping(cost, p,
                               minimizer_kwargs={'args': (ref, tgt)})
            if n <= 4:  # avoid basin-hopping in lower levels
                method = 'Powell'
        else:
            res = minimize(cost, p, args=(ref, tgt), method='Powell')
        p = res.x
        iter_callback(tgt, p)

    matrix = p_to_matrix(p)

    return matrix
