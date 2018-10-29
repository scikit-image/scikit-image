import numpy as np
from scipy.optimize import minimize
from scipy import ndimage as ndi

from .pyramids import pyramid_gaussian
from ..measure import compare_mse

__all__ = ['register_affine']


def _argmin_to_matrix(argmin):
    """
    Converts the optimisation parameters to a 3x3 transformation matrix

    The optimisation paramters are known as the argmin and are composed of
    the first 2 rows of the transformation matrix, as that is all that
    is used in an affine transformation.

    Parameters
    ----------
    argmin : array
        Input array giving the argument of the minimum function to optimise against

    Returns
    -------
    matrix : (3, 3) array
        A transformation matrix used to obtain a new image
    """

    matrix = np.empty((3, 3))

    matrix[0][0] = argmin[0]
    matrix[0][1] = argmin[1]
    matrix[0][2] = argmin[2]
    matrix[1][0] = argmin[3]
    matrix[1][1] = argmin[4]
    matrix[1][2] = argmin[5]
    matrix[2][0] = 0
    matrix[2][1] = 0
    matrix[2][2] = 1

    return matrix


def _matrix_to_argmin(matrix):
    """
    Converts a 3x3 transformation matrix to the optimisation parameters

    See the inverse function `_argmin_to_matrix`.

    Parameters
    ----------
    matrix : (3, 3) array
        A transformation matrix used to obtain a new image

    Returns
    -------
    argmin : array
        Output array giving the argument of the minimum function to optimise against

    """

    return (matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][0], matrix[1][1], matrix[1][2])


def register_affine(reference, target, *, cost=compare_mse, nlevels=None,
                    iter_callback=lambda img, matrix: None):
    """
    Returns a matrix which registers the target image to the reference image

    Uses image registration techniques, and only optimises with affine transformations.

    Parameters
    ----------
    reference : ndimage
        A reference image to compare against the target

    target : ndimage
        Our target for registration. Transforming this image using the return value
        `matrix` aligns it with the reference.

    cost : function, optional
        A cost function which takes two images and returns a score which is at a
        minimum when images are aligned. Uses the mean square error as default.

    nlevels : integer, optional
        Change the maximum height we use for creating the Gaussian pyramid.
        By default we take a guess based on the resolution of the image,
        as extremely low resolution images may hinder registration.

    iter_callback : function, optional
        If given, this function is called once per pyramid level with the current
        downsampled image and transformation matrix guess as the only arguments.
        This is useful for debugging or for plotting intermediate results
        during the iterative processs.

    Returns
    -------
    matrix : array
        A transformation matrix used to obtain a new image.
        ndi.affine_transform(target, matrix) will align your target image.

    """

    if nlevels is None:
        min_dim = min(reference.shape)
        max_level = max(int(np.log2([min_dim])[0]) - 2, 2)
        nlevels = min(max_level, 7)

    pyramid_ref = pyramid_gaussian(reference, max_layer=nlevels - 1)
    pyramid_tgt = pyramid_gaussian(target, max_layer=nlevels - 1)
    image_pairs = reversed(list(zip(pyramid_ref, pyramid_tgt)))
    argmin = np.zeros(6)
    argmin[0] = 1
    argmin[4] = 1

    for (ref, tgt) in image_pairs:
        def _cost(param):
            transformation = _argmin_to_matrix(param)
            transformed = ndi.affine_transform(tgt, transformation, order=1)
            return cost(ref, transformed)

        result = minimize(_cost, argmin, method='Powell')
        argmin = result.x
        iter_callback(tgt, _argmin_to_matrix(argmin))

    matrix = _argmin_to_matrix(argmin)

    return matrix
