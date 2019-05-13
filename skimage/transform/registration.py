import numpy as np
from scipy.optimize import minimize
from scipy import ndimage as ndi

from .pyramids import pyramid_gaussian
from ..measure import compare_nmi

__all__ = ['register_affine']


def _parameter_vector_to_matrix(parameter_vector, N):
    """
    Converts the optimisation parameters to a 3x3 transformation matrix

    The optimisation paramters are known as the parameter_vector and are
    composed of the first 2 rows of the transformation matrix, as that is
    all that is used in an affine transformation.

    Parameters
    ----------
    parameter_vector : (N*(N+1)) array
        Input array giving the argument of the minimum function to
        optimise against

    Returns
    -------
    matrix : (N+1, N+1) array
        A transformation matrix used to obtain a new image
    """

    return np.concatenate(
        (np.reshape(parameter_vector, (N, N + 1)), [[0] * N + [1]]), axis=0)


def _matrix_to_parameter_vector(matrix):
    """
    Converts a (N+1)x(N+1) transformation matrix to the optimisation parameters

    See the inverse function `_parameter_vector_to_matrix`.

    Parameters
    ----------
    matrix : (N+1, N+1) array
        A transformation matrix used to obtain a new image

    Returns
    -------
    parameter_vector : (N*(N+1)) array
        Output array giving the argument of the minimum function to
        optimise against

    """

    return matrix[:-1, :].ravel()


def cost_nmi(image0, image1, *, bins=100):
    """Negative of the normalized mutual information.

    See :func:`skimage.measure.compare_nmi` for more info.

    Parameters
    ----------
    image0, image1 : array
        The images to be compared. They should have the same shape.
    bins : int or sequence of int, optional
        The granularity of the histogram with which to compare the images.
        If it's a sequence, each number is the number of bins for that image.

    Returns
    -------
    cnmi : float
        The negative of the normalized mutual information between ``image0``
        and ``image1``.
    """
    return -compare_nmi(image0, image1, bins=bins)


def register_affine(reference, target, *, cost=cost_nmi, nlevels=None,
                    multichannel=False, inverse=True,
                    level_callback=lambda x: None):
    """
    Returns a matrix which registers the target image to the reference image

    Will only return an affine transformation matrix.

    Parameters
    ----------
    reference : ndimage
        A reference image to compare against the target
    target : ndimage
        Our target for registration. Transforming this image using the
        return value `matrix` aligns it with the reference.
    cost : function, optional
        A cost function which takes two images and returns a score which is
        at a minimum when images are aligned. Uses the mean square error as
        default.
    nlevels : integer, optional
        Change the maximum height we use for creating the Gaussian pyramid.
        By default we take a guess based on the resolution of the image,
        as extremely low resolution images may hinder registration.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension. By default, this is False.
    inverse : bool, optional
        Whether to return the inverse transform, which converts coordinates
        in the reference space to coordinates in the target space. For
        technical reasons, this is the transform expected by
        ``scipy.ndimage.affine_transform`` to map the target image to the
        reference space.
    level_callback : callable, optional
        If given, this function is called once per pyramid level with a tuple
        containing the current downsampled image, transformation matrix, and
        cost as the argument. This is useful for debugging or for plotting
        intermediate results during the iterative processs.

    Returns
    -------
    matrix : array
        A transformation matrix used to obtain a new image.
        ndi.affine_transform(target, matrix) will align your target image.

    Example
    -------
    >>> from skimage.data import camera
    >>> reference_image = camera()
    >>> r = 0.42  # radians
    >>> c, s = np.cos(r), np.sin(r)
    >>> matrix_transform = np.array([[c, -s, 0], [s, c, 50], [0, 0, 1]])
    >>> target_image = ndi.affine_transform(reference_image, matrix_transform)
    >>> output_matrix = register_affine(reference_image, target_image)
    >>> registered_target = ndi.affine_transform(target_image, output_matrix)

    """

    ndim = reference.ndim

    if nlevels is None:
        min_dim = min(reference.shape)
        max_level = max(int(np.log2([min_dim])[0]) - 2, 2)
        nlevels = min(max_level, 7)

    pyramid_ref = pyramid_gaussian(reference, max_layer=nlevels - 1,
        multichannel=multichannel)
    pyramid_tgt = pyramid_gaussian(target, max_layer=nlevels - 1,
        multichannel=multichannel)
    image_pairs = reversed(list(zip(pyramid_ref, pyramid_tgt)))
    parameter_vector = _matrix_to_parameter_vector(np.identity(ndim + 1))

    for ref, tgt in image_pairs:
        def _cost(param):
            transformation = _parameter_vector_to_matrix(param, ndim)
            transformed = ndi.affine_transform(tgt, transformation, order=1)
            return cost(ref, transformed)

        result = minimize(_cost, parameter_vector, method='Powell')
        parameter_vector = result.x
        level_callback((tgt,
                        _parameter_vector_to_matrix(parameter_vector, ndim),
                        result.fun))

    matrix = _parameter_vector_to_matrix(parameter_vector, ndim)

    if not inverse:
        # estimated is already inverse, so we invert for forward transform
        matrix = np.linalg.inv(matrix)

    return matrix
