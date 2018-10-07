import numpy as np
from scipy.optimize import basinhopping, minimize

from .pyramids import pyramid_gaussian
from ._warps import warp, SimilarityTransform

__all__ = ['register_affine']


def _mse(img1, img2):
    """
    Finds the error between the overlapping area of two images
    
    Parameters
    ----------
    img1 : (M, N) ndarray
        Input image used for reference
    img2 : (M, N) ndarray
        Input image which is compared to img1
    
    Returns
    -------
    err: int
        Error in the form of the squared difference between pixels
    """
    dim0 = min(img1.shape[0],img2.shape[0])
    dim1 = min(img1.shape[1],img2.shape[1])
    return ((img1[:dim0,:dim1]-img2[:dim0,:dim1])**2).sum()


def _cost_mse(param, reference_image, target_image):
    """
    Finds the error between the overlapping area of two images after transformations are done to one of them
    
    Parameters
    ----------
    param : array
        Input array giving the translation and rotation the target image will undergo
    reference_image : (M, N) ndarray
        Input image used for reference
    target_image : (M, N) ndarray
        Input image which is modified and then compared to reference_image
    
    Returns
    -------
    err: int
        Error in the form of the squared difference between pixels
    """
    transformation = p_to_matrix(param)
    transformed = warp(target_image, transformation, order=3)
    return _mse(reference_image, transformed)


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
    out = np.empty((3,3))
    
    out[0][0] = np.cos(r)
    out[0][1] = -1*np.sin(r)
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
    
    m = matrix.params
    return (np.arccos(m[0][0]), m[0][2], m[1][2])


def register_affine(reference, target, *, cost=_cost_mse, nlevels=7, method='Powell', iter_callback=lambda img, p: None):
    assert method in ['Powell', 'BH']
    
    pyramid_ref = reversed(pyramid_gaussian(reference, levels=6))
    pyramid_tgt = reversed(pyramid_gaussian(target, levels=6))
    levels = range(nlevels, 0, -1)
    image_pairs = zip(pyramid_ref, pyramid_tgt)
    p = np.zeros(3)

    for n, (ref, tgt) in zip(levels, image_pairs):
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
