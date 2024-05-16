"""Affine registration"""

import warnings

from functools import partial
from itertools import product, combinations_with_replacement, combinations

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize

from skimage.transform.pyramids import pyramid_gaussian
from skimage.metrics import normalized_mutual_information

from math import log, floor, cos, sin


def target_registration_error(shape, matrix):
    """
    Compute the displacement norm of the transform at at each pixel

    Parameters
    ----------
    shape : shape like
        Shape of the array
    matrix: ndarray
        Homogeneous matrix

    Returns
    -------
    TRE : ndarray
        Norm of the displacement given by the transform

    """
    # Create a regular set of points on the grid
    points = np.concatenate(
        [
            np.stack(
                [
                    x.flatten()
                    for x in np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
                ],
                axis=1,
            ).T,
            np.array([1] * np.prod(shape)).reshape(1, -1),
        ]
    )
    delta = matrix @ points - points
    TRE = np.linalg.norm(delta[: len(shape)], axis=0).reshape(shape)
    return TRE


def _parameter_vector_to_matrix(parameter, model, ndim):
    """
    Transform a vector of parameter to an affine matrix

    Parameters
    ----------
    parameter: ndarray
        Vector of parameters
    model: str
        Motion model 'affine', 'euclidean' or 'translation'
    ndim: int
        Image dimension (any dimensions)

    Returns
    -------
    Homogeneous matrix

    Raises
    ------
    NotImplementedError for unsupported motion models.

    Note
    ----
    The parameters are
    - translation : [dy,dx] in 2D or [dz,dy,dx] in 3D
    - euclidean : [dy,dx,theta] in 2D or [dz,dy,dx,alpha,beta,gamma] in 3D
    - affine: the top of the homogeneous matrix minus identity

    """

    # Test if the parameter is actually a homogeneous matrix already
    if parameter.shape == (ndim + 1, ndim + 1):
        return parameter

    if model.lower() == "translation":
        matrix = np.eye(ndim + 1)
        matrix[:ndim, -1] = parameter.ravel()
    elif model.lower() == "euclidean":
        matrix = np.eye(ndim + 1)
        # Rotations for each planes
        for k, a in enumerate(combinations(range(ndim), 2)):
            c, s = cos(parameter[ndim + k]), sin(parameter[ndim + k])
            R = np.eye(ndim + 1)
            R[a[0], a[0]] = c
            R[a[0], a[1]] = -s
            R[a[1], a[1]] = c
            R[a[1], a[0]] = s
            matrix = matrix @ R
        # Translation along each axis
        T = np.eye(ndim + 1)
        T[:ndim, -1] = parameter[:ndim].ravel()
        matrix = matrix @ T
        T = np.array([[1, 1, parameter[0]], [0, 1, parameter[1]], [0, 0, 1]])
    elif model.lower() == "affine":
        matrix = np.eye(ndim + 1, dtype=parameter.dtype)
        matrix[:ndim, -1] = parameter[:ndim].ravel()
        matrix[:ndim, :ndim] = parameter[ndim:].reshape(ndim, ndim) + np.eye(ndim)
    else:
        raise NotImplementedError(f"Model {model} is not supported")
    return matrix


def _scale_parameters(parameter, model, ndim, scale):
    """
    Scale the parameter vector or homogeneous matrix

    Parameters
    ----------
    parameter: ndarray
        Parameter vector or homogeneous matrix
    model: str
        Motion model 'affine', 'euclidean' or 'translation'
    ndim: int
        Dimensionality of the image
    scale: float
        Scaling factor

    Returns
    -------
    Scaled parameters as a ndarray

    Raises
    ------
    NotImplementedError for unsupported motion models.

    Note
    ----
    See _parameter_vector_to_matrix for the indices

    """
    scaled_parameters = parameter.copy()
    # Homogeneous matrix case
    if parameter.shape == (ndim + 1, ndim + 1):
        scaled_parameters[:ndim, -1] *= scale
    else:
        # Vector parameter
        if model.lower() == "translation":
            scaled_parameters *= scale
        elif model.lower() == "euclidean":
            # translation indices are assumed to be the first ndim elements
            scaled_parameters[:ndim] *= scale
        elif model.lower() == "affine":
            # Compute translation indices
            indices = np.arange((ndim + 1) * (ndim + 1)).reshape(ndim + 1, ndim + 1)[
                :ndim, -1
            ]
            scaled_parameters[indices] *= scale
        else:
            raise NotImplementedError(f"Model {model} is not supported")
    return scaled_parameters


def lucas_kanade_affine_solver(
    reference_image,
    moving_image,
    weights,
    channel_axis,
    matrix,
    model,
    *,
    max_iter=40,
    tol=1e-6,
):
    """
    Estimate affine motion between two images using a least square approach

    Parameters
    ----------
    reference_image : ndarray
        The firste image of the sequence.
    moving_image : ndarray
        The second image of the sequence.
    weights : ndarray
        Weights as an array with the same shape as reference_image
    channel_axis: int
        Index of the channel axis
    matrix : ndarray
        Initial homogeneous transformation matrix
    model : str
        Motion model 'affine', 'translation' or (2D) 'euclidean'
    max_iter : int
        Maximum number of inner iteration
    tol : float
        Tolerance of the norm of the update vector

    Returns
    -------
    matrix : ndarray
        The estimated homogenous transformation matrix

    Raises
    ------
    NotImplementedError for unsupported motion models.

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
        np.gradient(reference_image, axis=k) for k in range(1, reference_image.ndim)
    ]

    # Vector of gradients eg: [Iy Ix yIy xIy yIy yIx] (see eq. (27))
    elems = [*grad, *[x * dx for dx, x in product(grad, grid)]]

    # G matrix of eq. (32)
    G = np.zeros([len(elems)] * 2)
    for i, j in combinations_with_replacement(range(len(elems)), 2):
        G[i, j] = G[j, i] = (elems[i] * elems[j] * weights).sum()

    # Model reduction
    if model.lower() == "translation":
        H = np.eye(G.shape[0], ndim, dtype=np.float64)
    elif model.lower() == "euclidean":
        if ndim == 2:
            H = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                ],
            )
        else:
            raise NotImplementedError("Eulidean motion model implemented only in 2D")
    elif model.lower() == "affine":
        H = np.eye(ndim * (ndim + 1), dtype=matrix.dtype)
    else:
        raise NotImplementedError(f"Unsupported motion model {model}")

    for _ in range(max_iter):
        # Warp each channel
        moving_image_warp = np.stack(
            [ndi.affine_transform(plane, matrix) for plane in moving_image]
        )

        # Compute the error (eq. (26))
        error_image = reference_image - moving_image_warp

        # Compute b from eq. (33)
        b = np.array([[(e * error_image * weights).sum()] for e in elems])

        try:
            # Solve the system
            r = np.linalg.solve(H.T @ G @ H, H.T @ b)
            # update the matrix
            matrix = matrix @ _parameter_vector_to_matrix(r.ravel(), model, ndim)

            if np.linalg.norm(r) < tol:
                break

        except np.linalg.LinAlgError:
            break

    return matrix


def cost_nmi(image0, image1, *, bins=100, weights=None):
    """
    Negative of the normalized mutual information.

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

    return -normalized_mutual_information(
        image0.squeeze(), image1.squeeze(), bins=bins, weights=weights
    )


def _param_cost(
    parameters,
    reference_image,
    moving_image,
    weights,
    *,
    cost=cost_nmi,
    model="affine",
):
    """
    Compute the registration cost for the current parameters

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
    cost: float
        Evaluated cost

    """

    ndim = reference_image.ndim - 1

    matrix = _parameter_vector_to_matrix(parameters, model, ndim)

    moving_image_warp = np.stack(
        [ndi.affine_transform(plane, matrix) for plane in moving_image]
    )

    return cost(reference_image, moving_image_warp, weights=weights)


def studholme_affine_solver(
    reference_image,
    moving_image,
    weights,
    channel_axis,
    matrix,
    model,
    *,
    method="Powell",
    options={"maxiter": 10, "disp": False},
    cost=cost_nmi,
):
    """
    Solver maximizing mutual information using Powell's method

    Parameters
    ----------
    reference_image : ndarray
        The first image of the sequence.
    moving_image : ndarray
        The second image of the sequence.
    weights: ndarray
        Weights or mask
    channel_axis: int | None
        Index of the channel axis
    matrix: ndarray
        Initial value of the transform, here matrix are parameters
    model: str
        Motion model: "affine", "translation" or "euclidean"
    method: str
        Minimization method for scipy.optimization.minimize
    options: dict
        options for scipy.optimization.minimize
    cost: function
        Cost function minimize
    vector_to_matrix: function(param, ndim) -> ndarray
        Convert a vector of parameters to a matrix

    Returns
    -------
    matrix:
        Homogeneous transform matrix

    Raises
    ------
    NotImplementedError for unsupported motion models.

    Reference
    ---------
    .. [1] Studholme C, Hill DL, Hawkes DJ. Automated 3-D registration of MR
        and CT images of the head. Med Image Anal. 1996 Jun;1(2):163-75.
    .. [2] J. Nunez-Iglesias, S. van der Walt, and H. Dashnow, Elegant SciPy:
        The Art of Scientific Python. O’Reilly Media, Inc., 2017.

    """

    if channel_axis is not None:
        reference_image = np.moveaxis(reference_image, channel_axis, 0)
        moving_image = np.moveaxis(moving_image, channel_axis, 0)
    else:
        reference_image = np.expand_dims(reference_image, 0)
        moving_image = np.expand_dims(moving_image, 0)

    ndim = reference_image.ndim - 1

    if matrix is None:
        if model.lower() == "translation":
            matrix = np.zeros(ndim)
        elif model.lower() == "euclidean":
            nrotations = len([x for x in combinations(range(ndim), 2)])
            matrix = np.zeros(ndim + nrotations)
        elif model.lower() == "affine":
            matrix = np.zeros(ndim * (ndim + 1))
        else:
            raise NotImplementedError(f"Motion model {model} not implemented.")

    cost = partial(
        _param_cost,
        reference_image=reference_image,
        moving_image=moving_image,
        weights=weights,
        cost=cost,
        model=model,
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
    model="affine",
    solver=lucas_kanade_affine_solver,
    pyramid_downscale=2,
    pyramid_minimum_size=32,
):
    """
    Coarse-to-fine affine motion estimation between two images

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
    model: str
        Motion model
    solver: lambda(reference_image, moving_image, weights, channel_axis, matrix) -> matrix
        Affine motion solver can be lucas_kanade_affine_solver or studholme_affine_solver
    pyramid_scale : float
        The pyramid downscale factor.
    pyramid_minimum_size : int
        The minimum size for any dimension of the pyramid levels.

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
    >>> reference = data.camera()
    >>> r = -0.12  # radians
    >>> c, s = np.cos(r), np.sin(r)
    >>> transform = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    >>> moving = ndi.affine_transform(reference, transform)
    >>> matrix = affine(reference, moving)
    >>> registered = ndi.affine_transform(moving, matrix)

    """

    ndim = reference_image.ndim if channel_axis is None else reference_image.ndim - 1

    if channel_axis is not None:
        shape = [d for k, d in enumerate(reference_image.shape) if k != channel_axis]
    else:
        shape = reference_image.shape
        if min(shape) <= 6:
            warnings.warn(f"No channel axis specified for shape {shape}")

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
        model=model,
    )

    for J0, J1, W in pyramid[1:]:
        matrix = _scale_parameters(matrix, model, ndim, pyramid_downscale)
        matrix = solver(
            J0, J1, W, channel_axis=channel_axis, matrix=matrix, model=model
        )

    return _parameter_vector_to_matrix(matrix, model, ndim)
