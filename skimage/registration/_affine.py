import warnings
from functools import partial
from itertools import product, combinations_with_replacement

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize

from skimage.transform.pyramids import pyramid_gaussian
from skimage.metrics import normalized_mutual_information

from math import log, floor


def lucas_kanade_affine_solver(
    reference_image,
    moving_image,
    weights,
    channel_axis,
    matrix,
    *,
    max_iter=40,
    tol=1e-6,
    warp=ndi.affine_transform,
    vector_to_matrix=None,
):
    """Estimate affine motion between two images using a least square approach

    Parameters
    ----------

    reference_image : ndarray
        The first gray scale image of the sequence.
    moving_image : ndarray
        The second gray scale image of the sequence.
    weights : ndarray
        Weights as an array with the same shape as reference_image
    channel_axis: int
        Index of the channel axis
    matrix : ndarray
        Initial homogeneous transformation matrix
    max_iter : int
        Number of inner iteration
    tol : float
        Tolerance of the norm of the update vector
    warp:
        Affine transform funciton ndi.affine_transform with ij conventions

    Returns
    -------
    matrix : ndarray
        The estimated transformation matrix

    Reference
    ---------
    .. [1] http://robots.stanford.edu/cs223b04/algo_affine_tracking.pdf

    """

    if weights is None:
        weights = 1.0

    if channel_axis is not None:
        reference_image = np.moveaxis(reference_image, channel_axis, 0)
        moving_image = np.moveaxis(moving_image, channel_axis, 0)
    else:
        reference_image = np.expand_dims(reference_image, 0)
        moving_image = np.expand_dims(moving_image, 0)

    # We assume that there is always a channel axis in the 1st dimension
    ndim = reference_image.ndim - 1

    # Initialize with the identity
    if matrix is None:
        matrix = np.eye(ndim + 1, dtype=np.float64)

    # Compute the ij grids along each non channel axis
    grid = np.meshgrid(
        *[np.arange(n) for n in reference_image.shape[1:]], indexing="ij"
    )

    # Compute the 1st derivative of the image in each non channel axis
    grad = [
        np.gradient(reference_image, axis=k)[0] for k in range(1, reference_image.ndim)
    ]

    # Vector of gradients eg: [Iy Ix yIy xIy yIy yIx] (see eq. (27))
    elems = [*grad, *[x * dx for dx, x in product(grad, grid)]]

    # G matrix of eq. (32)
    G = np.zeros([len(elems)] * 2)
    for i, j in combinations_with_replacement(range(len(elems)), 2):
        G[i, j] = G[j, i] = (elems[i] * elems[j] * weights).sum()

    Id = np.eye(ndim, dtype=matrix.dtype)

    for _ in range(max_iter):
        # Warp each channel
        moving_image_warp = np.stack([warp(plane, matrix) for plane in moving_image])

        # Compute the error (eq. (26))
        error_image = reference_image - moving_image_warp

        # Compute b from eq. (33)
        b = np.array([[(e * error_image * weights).sum()] for e in elems])

        try:
            # Solve the system
            r = np.linalg.solve(G, b)

            # update the current matrix (eq. (36-37))
            matrix[:ndim, -1] += (matrix[:ndim, :ndim] @ r[:ndim]).ravel()
            matrix[:ndim, :ndim] @= r[ndim:].reshape(ndim, ndim) + Id

            if np.linalg.norm(r) < tol:
                break

        except np.linalg.LinAlgError:
            break

    return matrix


def _parameter_vector_to_matrix(parameters, ndim):
    matrix = np.eye(ndim + 1, dtype=np.float64)
    matrix[:ndim, :] = parameters.reshape(ndim, ndim + 1)
    return matrix


def cost_nmi(image0, image1, *, bins=100, weights=None):
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

    return -normalized_mutual_information(image0, image1, bins=bins, weights=weights)


def _param_cost(
    parameters,
    reference_image,
    moving_image,
    weights,
    *,
    cost=cost_nmi,
    vector_to_matrix=_parameter_vector_to_matrix,
    warp=ndi.affine_transform,
):
    """Compute the registration cost for the current parameters

    Parameters
    ----------
    parameters: ndarray
        Parameters of the transform
    reference_image: ndarray
        Reference image
    moving_image: ndarray
        Moving image
    weights: ndarray | None
        Weights
    cost: function
        Cost between registered image and reference
    vector_to_matrix:
        Convert the vector parameter to the homogeneous matrix

    Returns
    -------
    Evaluated cost
    """

    ndim = reference_image.ndim - 1

    matrix = vector_to_matrix(parameters, ndim)

    moving_image_warp = np.stack([warp(plane, matrix) for plane in moving_image])

    return cost(reference_image, moving_image_warp, weights=weights)


def studholme_affine_solver(
    reference_image,
    moving_image,
    weights,
    channel_axis,
    matrix,
    *,
    method="Powell",
    options={"maxiter": 10, "disp": False},
    cost=cost_nmi,
    vector_to_matrix=_parameter_vector_to_matrix,
    warp=ndi.affine_transform,
):
    """Solver maximizing mutual information using Powell's method

    Parameters
    ----------
    reference_image : ndarray
        The first gray scale image of the sequence.
    moving_image : ndarray
        The second gray scale image of the sequence.
    weights: ndarray
        Weights or mask
    channel_axis: int | None
        Index of the channel axis
    matrix: ndarray
        Initial value of the transform
    options: dict
        options for scipy.optimization.minimize("Powell")
    cost: function
        Cost function minimize
    vector_to_matrix: function(param, ndim) -> ndarray
        Convert a vector of parameters to a matrix

    Returns
    -------
    matrix:
        Homogeneous transform matrix

    Reference
    ---------
    .. [1] Studholme C, Hill DL, Hawkes DJ. Automated 3-D registration of MR
        and CT images of the head. Med Image Anal. 1996 Jun;1(2):163-75.
    .. [2] J. Nunez-Iglesias, S. van der Walt, and H. Dashnow, Elegant SciPy:
        The Art of Scientific Python. O’Reilly Media, Inc., 2017.

    Note:
    from PR #3544
    """

    if channel_axis is not None:
        reference_image = np.moveaxis(reference_image, channel_axis, 0)
        moving_image = np.moveaxis(moving_image, channel_axis, 0)
    else:
        reference_image = np.expand_dims(reference_image, 0)
        moving_image = np.expand_dims(moving_image, 0)

    ndim = reference_image.ndim - 1

    if matrix is None:
        matrix = np.eye(ndim, ndim + 1, dtype=np.float64).flatten()

    cost = partial(
        _param_cost,
        reference_image=reference_image,
        moving_image=moving_image,
        weights=weights,
        cost=cost,
        vector_to_matrix=vector_to_matrix,
        warp=warp,
    )

    result = minimize(cost, x0=matrix, method=method, options=options)

    return result.x


def affine(
    reference_image,
    moving_image,
    *,
    weights=None,
    channel_axis=None,
    matrix=None,
    solver=lucas_kanade_affine_solver,
    pyramid_downscale=2,
    pyramid_minimum_size=32,
    translation_indices=None,
    vector_to_matrix=None,
):
    """Coarse-to-fine affine motion estimation between two images

    Parameters
    ----------
    reference_image : ndarray
        The first gray scale image of the sequence.
    moving_image : ndarray
        The second gray scale image of the sequence.
    weights : ndarray | None
        The weights array with same shape as reference_image
    channel_axis: int | None
        Index of the channel axis
    matrix : ndarray
        Intial guess for the homogeneous transformation matrix
    solver: lambda(reference_image, moving_image, weights, channel_axis, matrix) -> matrix
        Affine motion solver can be lucas_kanade_affine_solver or studholme_affine_solver
    pyramid_scale : float
        The pyramid downscale factor.
    pyramid_minimum_size : int
        The minimum size for any dimension of the pyramid levels.
    translation_indices:
        Indices of the translation parameters that are not scale invariant
    Returns
    -------
    matrix: ndarray
        Transformation matrix

    Note
    ----
    Generate image pyramids and apply the solver at each level passing the
    previous affine parameters from the previous estimate.

    The estimated matrix can be used with scikit.ndimage.affine to register the moving image
    to the reference image.

    Reference
    ---------
    .. [1] J. Nunez-Iglesias, S. van der Walt, and H. Dashnow, Elegant SciPy:
    The Art of Scientific Python. O’Reilly Media, Inc., 2017.

    Example
    -------
    >>> from skimage import data
    >>> import numpy as np
    >>> from scipy import ndimage as ndi
    >>> from skimage.registration import affine
    >>> import matplotlib.pyplot as plt
    >>> reference = data.astronaut()[...,0]
    >>> r = -0.12  # radians
    >>> c, s = np.cos(r), np.sin(r)
    >>> matrix_transform = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    >>> moving_image = ndi.affine_transform(reference_image, matrix_transform)
    >>> matrix = affine(reference_image, moving_image)
    >>> registered_moving = ndi.affine_transform(moving_image, matrix)
    >>> plt.imshow(registered_moving)

    """

    ndim = reference_image.ndim if channel_axis is None else reference_image.ndim - 1

    if channel_axis is not None:
        shape = [d for k, d in enumerate(reference_image.shape) if k != channel_axis]
    else:
        shape = reference_image.shape
        if min(shape) <= 6:
            warnings.warn(f"No channel axis specified for shape {shape}")

    if translation_indices is None:
        translation_indices = np.arange((ndim + 1) * (ndim + 1)).reshape(
            ndim + 1, ndim + 1
        )[:ndim, -1]

    if vector_to_matrix is None:
        if solver is lucas_kanade_affine_solver:

            def vector_to_matrix(x, ndim):
                return x

        elif solver is studholme_affine_solver:
            vector_to_matrix = _parameter_vector_to_matrix

    max_layer = int(
        floor(
            log(min(shape), pyramid_downscale)
            - log(pyramid_minimum_size, pyramid_downscale)
        )
    )

    pyramid = list(
        zip(
            *[
                reversed(
                    list(
                        pyramid_gaussian(
                            x,
                            max_layer=max_layer,
                            downscale=pyramid_downscale,
                            preserve_range=True,
                            channel_axis=channel_axis,
                        )
                        if x is not None
                        else [None] * (max_layer + 1)
                    )
                )
                for x in [reference_image, moving_image, weights]
            ]
        )
    )

    matrix = solver(
        pyramid[0][0],
        pyramid[0][1],
        pyramid[0][2],
        channel_axis=channel_axis,
        matrix=matrix,
        vector_to_matrix=vector_to_matrix,
    )

    for J0, J1, W in pyramid[1:]:
        matrix.ravel()[translation_indices] *= pyramid_downscale
        matrix = solver(
            J0,
            J1,
            W,
            channel_axis=channel_axis,
            matrix=matrix,
            vector_to_matrix=vector_to_matrix,
        )

    return vector_to_matrix(matrix, ndim)
