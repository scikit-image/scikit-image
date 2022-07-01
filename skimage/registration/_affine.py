import functools
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize

from skimage.transform.pyramids import pyramid_gaussian
from skimage.metrics import normalized_mutual_information

__all__ = ['affine']


def _parameter_vector_to_matrix(parameter_vector):
    """Convert m optimization parameters to a (n+1, n+1) transformation matrix.

    By default (the case of this function), the parameter vector is taken to
    be the first n rows of the affine transformation matrix in homogeneous
    coordinate space.

    Parameters
    ----------
    parameter_vector : (ndim*(ndim+1)) array
        A vector of M = N * (N+1) parameters.

    Returns
    -------
    matrix : (ndim+1, ndim+1) array
        A transformation matrix used to affine-map coordinates in an
        ``ndim``-dimensional space.
    """
    m = parameter_vector.shape[0]
    ndim = int((np.sqrt(4*m + 1) - 1) / 2)
    top_matrix = np.reshape(parameter_vector, (ndim, ndim+1))
    bottom_row = np.array([[0] * ndim + [1]])
    return np.concatenate((top_matrix, bottom_row), axis=0)


def cost_nmi(image0, image1, *, bins=100):
    """Negative of the normalized mutual information.

    See :func:`skimage.metrics.normalized_mutual_information` for more info.

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
    return -normalized_mutual_information(image0, image1, bins=bins)


def _param_cost(reference_image, moving_image, parameter_vector, *,
                vector_to_matrix, cost, multichannel):
    transformation = vector_to_matrix(parameter_vector)
    if not multichannel:
        transformed = ndi.affine_transform(moving_image, transformation,
                                           order=1)
    else:
        transformed = np.zeros_like(moving_image)
        for ch in range(moving_image.shape[-1]):
            ndi.affine_transform(moving_image[..., ch], transformation,
                                 order=1, output=transformed[..., ch])
    return cost(reference_image, transformed)


def affine(reference_image,
           moving_image,
           *,
           cost=cost_nmi,
           method='Powell',
           initial_parameters=None,
           pyramid_scale=2,
           pyramid_minimum_size=8,
           level_callback=None,
           inverse=True,
           channel_axis=None,
           vector_to_matrix=None,
           translation_indices=None,
           **kwargs):
    """Find a transformation matrix to register a moving image to a reference.

    Parameters
    ----------
    reference_image : ndarray
        A reference image to compare against the target.
    moving_image : ndarray
        Our target for registration. Transforming this image using the
        returned matrix aligns it with the reference.
    cost : function, optional
        A cost function which takes two images and returns a score which is
        at a minimum when images are aligned. Uses the normalized mutual
        information by default.
    initial_parameters : array of float, optional
        The initial vector to optimize. This vector should have the same
        dimensionality as the transform being optimized. For example, a 2D
        affine transform has 6 parameters. A 2D rigid transform, on the other
        hand, only has 3 parameters.
    vector_to_matrix : callable, array (M,) -> array-like (N+1, N+1), optional
        A function to convert a linear vector of parameters, as used by
        `scipy.optimize.minimize`, to an affine transformation matrix in
        homogeneous coordinates.
    translation_indices : array of int, optional
        The location of the translation parameters in the parameter vector. If
        None, the positions of the translation parameters in the raveled
        affine transformation matrix, in homogeneous coordinates, are used. For
        example, in a 2D transform, the translation parameters are in the
        top two positions of the third column of the 3 x 3 matrix, which
        corresponds to the linear indices [2, 5].
        The translation parameters are special in this class of transforms
        because they are the only ones not scale-invariant. This means that
        they need to be adjusted for each level of the image pyramid.
    inverse : bool, optional
        Whether to return the inverse transform, which converts coordinates
        in the reference space to coordinates in the target space. For
        technical reasons, this is the transform expected by
        ``scipy.ndimage.affine_transform`` to map the target image to the
        reference space. Defaults to True.
    pyramid_scale : float, optional
        Scaling factor to generate the image pyramid. The affine transformation
        is estimated first for a downscaled version of the image, then
        progressively refined with higher resolutions. This parameter controls
        the increase in resolution at each level.
    pyramid_minimum_size : integer, optional
        The smallest size for an image along any dimension. This value
        determines the size of the image pyramid used. Choosing a smaller value
        here can cause registration errors, but a larger value could speed up
        registration when the alignment is easy.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension. By default, this is False.
    level_callback : callable, optional
        If given, this function is called once per pyramid level with a tuple
        containing the current downsampled image, transformation matrix, and
        cost as the argument. This is useful for debugging or for plotting
        intermediate results during the iterative process.
    method : string or callable
        Method of minimization.  See ``scipy.optimize.minimize`` for available
        options.
    **kwargs : keyword arguments
        Keyword arguments passed through to ``scipy.optimize.minimize``

    Returns
    -------
    matrix : array, or object coercible to array
        A transformation matrix used to obtain a new image.
        ``ndi.affine_transform(moving, matrix)`` will align the moving image to
        the reference.

    Example
    -------
    >>> from skimage.data import astronaut
    >>> reference_image = astronaut()[..., 1]
    >>> r = -0.12  # radians
    >>> c, s = np.cos(r), np.sin(r)
    >>> matrix_transform = np.array([[c, -s, 0], [s, c, 50], [0, 0, 1]])
    >>> moving_image = ndi.affine_transform(reference_image, matrix_transform)
    >>> matrix = affine(reference_image, moving_image)
    >>> registered_moving = ndi.affine_transform(moving_image, matrix)
    """

    # ignore the channels if present
    ndim = reference_image.ndim if not multichannel else reference_image.ndim - 1
    if ndim == 0:
        raise ValueError(
            'Input images must have at least 1 spatial dimension.'
        )

    min_dim = min(reference_image.shape[:ndim])
    nlevels = int(np.floor(np.log2(min_dim) - np.log2(pyramid_minimum_size)))

    pyramid_ref = pyramid_gaussian(reference_image, downscale=pyramid_scale,
                                   max_layer=nlevels,
                                   channel_axis=-1)
    pyramid_mvg = pyramid_gaussian(moving_image, downscale=pyramid_scale,
                                   max_layer=nlevels,
                                   channel_axis=-1)
    image_pairs = reversed(list(zip(pyramid_ref, pyramid_mvg)))

    if initial_parameters is None:
        initial_parameters = np.eye(ndim, ndim + 1).ravel()
    parameter_vector = initial_parameters
    parameter_vector[translation_indices] /= pyramid_scale ** (nlevels + 1)

    if vector_to_matrix is None:
        vector_to_matrix = _parameter_vector_to_matrix

    if translation_indices is None:
        translation_indices = slice(ndim, ndim**2 - 1, ndim)

    for ref, mvg in image_pairs:
        parameter_vector[translation_indices] *= pyramid_scale
        _cost = functools.partial(_param_cost, ref, mvg,
                                  vector_to_matrix=vector_to_matrix,
                                  cost=cost, multichannel=multichannel)
        result = minimize(_cost, x0=parameter_vector, method=method, **kwargs)
        parameter_vector = result.x
        if level_callback is not None:
            level_callback(
                (mvg,
                 vector_to_matrix(parameter_vector),
                 result.fun)
            )

    matrix = vector_to_matrix(parameter_vector)

    if not inverse:
        # estimated is already inverse, so we invert for forward transform
        matrix = np.linalg.inv(matrix)

    return matrix
