import numpy as np
from scipy.optimize import minimize
from scipy import ndimage as ndi

from .pyramids import pyramid_gaussian
from ..measure import compare_mse

__all__ = ['register_affine']


def _parameter_vector_to_matrix(parameter_vector):
    """
    Converts the optimisation parameters to a 3x3 transformation matrix

    The optimisation paramters are known as the parameter_vector and are composed of
    the first 2 rows of the transformation matrix, as that is all that
    is used in an affine transformation.

    Parameters
    ----------
    parameter_vector : (6) array
        Input array giving the argument of the minimum function to optimise against

    Returns
    -------
    matrix : (3, 3) array
        A transformation matrix used to obtain a new image
    """

    return np.concatenate((np.reshape(parameter_vector, (2, 3)), [[0, 0, 1]]), axis=0)


def _matrix_to_parameter_vector(matrix):
    """
    Converts a 3x3 transformation matrix to the optimisation parameters

    See the inverse function `_parameter_vector_to_matrix`.

    Parameters
    ----------
    matrix : (3, 3) array
        A transformation matrix used to obtain a new image

    Returns
    -------
    parameter_vector : (6) array
        Output array giving the argument of the minimum function to optimise against

    """

    return matrix[:2, :].ravel()


def register_affine(reference, target, *, cost=compare_mse, nlevels=None,
                    iter_callback=lambda img, matrix: None):
    """
    Returns a matrix which registers the target image to the reference image

    Will only return an affine transformation matrix.

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

    Example
    -------
    >>> reference_image = camera()
    >>> matrix_transform = [
            [np.cos(0.12), -np.sin(0.12), 0.2],
            [np.sin(0.12),  np.cos(0.12), 0.1],
            [0,             0,            1.0]]
    >>> target_image = ndi.affine_transform(reference_image, matrix_transform)
    >>> output_image = register_affine(reference_image, target_image)

    """

    if nlevels is None:
        min_dim = min(reference.shape)
        max_level = max(int(np.log2([min_dim])[0]) - 2, 2)
        nlevels = min(max_level, 7)

    pyramid_ref = pyramid_gaussian(reference, max_layer=nlevels - 1)
    pyramid_tgt = pyramid_gaussian(target, max_layer=nlevels - 1)
    image_pairs = reversed(list(zip(pyramid_ref, pyramid_tgt)))
    parameter_vector = np.zeros(6)
    parameter_vector[0] = 1
    parameter_vector[4] = 1

    for (ref, tgt) in image_pairs:
        def _cost(param):
            transformation = _parameter_vector_to_matrix(param)
            transformed = ndi.affine_transform(tgt, transformation, order=1)
            return cost(ref, transformed)

        result = minimize(_cost, parameter_vector, method='Powell')
        parameter_vector = result.x
        iter_callback(tgt, _parameter_vector_to_matrix(parameter_vector))

    matrix = _parameter_vector_to_matrix(parameter_vector)

    return matrix
