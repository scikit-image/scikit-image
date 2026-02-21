"""Affine registration"""

import warnings
from functools import partial
from itertools import combinations, combinations_with_replacement, product
from math import cos, floor, log, pow, sin

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize

from skimage.metrics import normalized_mutual_information
from skimage.transform.pyramids import pyramid_gaussian


def target_registration_error(shape, matrix):
    """
    Compute the displacement norm of the transform at each pixel.

    Parameters
    ----------
    shape : tuple of (int, ...)
        Shape of the array.
    matrix : ndarray
        Homogeneous matrix.

    Returns
    -------
    error : ndarray
        Norm of the displacement given by the transform.
    """
    # Create a regular set of points on the grid
    slc = [slice(0, n) for n in shape]
    N = np.prod(shape)
    points = np.stack([*[x.ravel() for x in np.mgrid[slc]], np.ones(N)])
    # compute the displacement
    delta = matrix @ points - points
    error = np.linalg.norm(delta[: len(shape)], axis=0).reshape(shape)
    return error


def _parameter_vector_to_matrix(parameters, model, ndim):
    """
    Transform a vector of parameters into an affine matrix.

    Parameters
    ----------
    parameters : ndarray
        Vector of parameters (see note).
    model : {'affine', 'euclidean', 'translation'}
        Motion model 'affine', 'euclidean' or 'translation'.
    ndim : int
        Image dimensionality.

    Returns
    -------
    matrix : ndarray
        Homogeneous matrix.

    Raises
    ------
    NotImplementedError
        For unsupported motion models.

    Note
    ----
    The parameters are:
    - translation : [dy,dx] in 2D or [dz,dy,dx] in 3D,
    - euclidean : [dy,dx,theta] in 2D or [dz,dy,dx,alpha,beta,gamma] in 3D,
    - affine: the top of the homogeneous matrix minus identity.
    """

    # Test if the parameter is actually a homogeneous matrix already
    if parameters.shape == (ndim + 1, ndim + 1):
        return parameters

    if model.lower() == "translation":
        matrix = np.eye(ndim + 1)
        matrix[:ndim, -1] = parameters.ravel()
    elif model.lower() == "euclidean":
        matrix = np.eye(ndim + 1)
        # Rotations for each planes
        for k, a in enumerate(combinations(range(ndim), 2)):
            c, s = cos(parameters[ndim + k]), sin(parameters[ndim + k])
            rot = np.eye(ndim + 1)
            rot[a[0], a[0]] = c
            rot[a[0], a[1]] = -s
            rot[a[1], a[1]] = c
            rot[a[1], a[0]] = s
            matrix = matrix @ rot
        # Translation along each axis
        trans = np.eye(ndim + 1)
        trans[:ndim, -1] = parameters[:ndim].ravel()
        matrix = matrix @ trans
    elif model.lower() == "affine":
        matrix = np.eye(ndim + 1, dtype=parameters.dtype)
        matrix[:ndim, -1] = parameters[:ndim].ravel()
        matrix[:ndim, :ndim] = parameters[ndim:].reshape(ndim, ndim) + np.eye(ndim)
    else:
        raise NotImplementedError(f"Model {model} is not supported")
    return matrix


def _matrix_to_parameter_vector(matrix, model):
    """
    Transform a vector of parameters into an affine matrix.

    Parameters
    ----------
    matrix : ndarray
        Homogeneous matrix.
    model : {'affine', 'euclidean', 'translation'}
        Motion model 'affine', 'euclidean' or 'translation'.

    Returns
    -------
    parameters : ndarray
        Vector of parameters (see note).

    Raises
    ------
    NotImplementedError
        For unsupported motion models.

    Note
    ----
    The parameters are:
    - translation : [dy,dx] in 2D or [dz,dy,dx] in 3D,
    - euclidean : [dy,dx,angle] in 2D or [dz,dy,dx,yaw,pitch,roll] in 3D,
    - affine: the top of the homogeneous matrix minus identity.
    """

    ndim = matrix.shape[0] - 1

    if model.lower() == "translation":
        parameters = matrix[:ndim, -1].ravel()
    elif model.lower() == "euclidean":
        parameters = np.zeros(ndim + len(list(combinations(range(ndim), 2))))
        # Rotations
        if ndim == 2:
            parameters[ndim] = np.arctan2(matrix[1, 0], matrix[0, 0])
        elif ndim == 3:
            R = matrix[:3, :3]
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            if sy < 1e-6:  # not a gimbal lock
                parameters[ndim] = np.arctan2(R[1, 0], R[0, 0])
                parameters[ndim + 1] = np.arctan2(-R[2, 0], sy)
                parameters[ndim + 2] = np.arctan2(R[2, 1], R[2, 2])
            else:
                parameters[ndim] = np.arctan2(R[1, 2], R[1, 1])
                parameters[ndim + 1] = np.arctan2(-R[2, 0], sy)
        else:
            raise NotImplementedError("Eulidean motion model implemented only in 2D/3D")
        # Translation along each axis
        parameters[:ndim] = (
            np.linalg.inv(matrix[:-1, :-1]) @ matrix[:ndim, -1]
        ).ravel()
    elif model.lower() == "affine":
        parameters = np.zeros(ndim * (ndim + 1))
        parameters[:ndim] = matrix[:ndim, -1].ravel()
        parameters[ndim:] = (matrix[:ndim, :ndim] - np.eye(ndim)).ravel()
    else:
        raise NotImplementedError(f"Model {model} is not supported")
    return parameters


def _scale_parameters(parameters, model, ndim, scale):
    """
    Scale the parameter vector or homogeneous matrix.

    Parameters
    ----------
    parameters : ndarray
        Vector of parameters or homogeneous matrix.
    model : {'affine', 'euclidean', 'translation'}
        Motion model 'affine', 'euclidean', or 'translation'.
    ndim : int
        Dimensionality of the image.
    scale : float
        Scaling factor.

    Returns
    -------
    vector : ndarray
        Vector of scaled parameters as a ndarray.

    Raises
    ------
    NotImplementedError
        For unsupported motion models.

    Note
    ----
    See _parameter_vector_to_matrix for the indices.
    """

    scaled_parameters = parameters.copy()

    # Homogeneous matrix case
    if parameters.shape == (ndim + 1, ndim + 1):
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


def solver_affine_lucas_kanade(
    reference_image,
    moving_image,
    *,
    weights,
    channel_axis,
    matrix,
    model,
    max_iter=40,
    tol=1e-6,
):
    """
    Estimate affine motion between two images using a linearized least square
    approach.

    Parameters
    ----------
    reference_image : ndarray
        The first image of the sequence.
    moving_image : ndarray
        The second image of the sequence.
    weights : ndarray
        Weights as an array with the same shape as reference_image.
    channel_axis : int
        Index of the channel axis.
    matrix : ndarray
        Initial homogeneous transformation matrix.
    model : {'affine', 'euclidean', 'translation'}
        Motion model 'affine', 'translation' or (2D only) 'euclidean'.
    max_iter : int
        Maximum number of inner iterations.
    tol : float
        Tolerance of the norm of the update vector.

    Returns
    -------
    matrix : ndarray
        The estimated homogeneous transformation matrix.

    Raises
    ------
    NotImplementedError
        For unsupported motion models.

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
        *[np.arange(n, dtype=np.float32) for n in reference_image.shape[1:]],
        indexing="ij",
    )

    # Compute the 1st derivative of the image in each non channel axis
    grad = [
        np.gradient(reference_image.astype(np.float32), axis=k)
        for k in range(1, reference_image.ndim)
    ]

    # Vector of gradients eg: [Iy Ix yIy xIy yIy yIx] (see eq. (27))
    elems = [*grad, *[x * dx for dx, x in product(grad, grid)]]

    # G matrix of eq. (32)
    g_mat = np.zeros([len(elems)] * 2)
    for i, j in combinations_with_replacement(range(len(elems)), 2):
        g_mat[i, j] = g_mat[j, i] = (elems[i] * elems[j] * weights).sum()

    # Model reduction
    if model.lower() == "translation":
        h_mat = np.eye(g_mat.shape[0], ndim, dtype=np.float64)
    elif model.lower() == "euclidean":
        if ndim == 2:
            h_mat = np.array(
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
        h_mat = np.eye(ndim * (ndim + 1), dtype=matrix.dtype)
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
            r = np.linalg.solve(h_mat.T @ g_mat @ h_mat, h_mat.T @ b)
            # update the matrix
            matrix = matrix @ _parameter_vector_to_matrix(r.ravel(), model, ndim)

            if np.linalg.norm(r) < tol:
                break

        except np.linalg.LinAlgError:
            warnings.warn("Failed to invert linear system (np.linalg.solve).")
            break

    return matrix


def _studholme_param_cost(
    parameters,
    reference_image,
    moving_image,
    weights,
    cost,
    *,
    model="affine",
):
    """
    Compute the registration cost for given parameters.

    Parameters
    ----------
    parameters : ndarray
        Parameters of the transform.
    reference_image : ndarray
        Reference image.
    moving_image : ndarray
        Moving image.
    weights : ndarray | None
        Weights.
    cost : Callable, lambda(reference_image, moving_image, weights) -> float
        Cost between registered image and reference image.
    model : {'affine', 'euclidean', 'translation'}
        Motion model: "affine", "translation" or "euclidean".

    Returns
    -------
    cost : float
        Evaluated cost.
    """

    ndim = reference_image.ndim - 1

    # scale translation to image size
    scale = np.max(reference_image.shape)
    parameters = _scale_parameters(parameters, model, ndim, scale)

    matrix = _parameter_vector_to_matrix(parameters, model, ndim)

    moving_image_warp = np.stack(
        [ndi.affine_transform(plane, matrix) for plane in moving_image]
    )

    return cost(reference_image, moving_image_warp, weights)


def solver_affine_studholme(
    reference_image,
    moving_image,
    weights,
    channel_axis,
    matrix,
    model,
    *,
    method="Powell",
    options={"maxiter": 30, "disp": False},
    cost=lambda im0, im1, w: -normalized_mutual_information(
        im0.squeeze(), im1.squeeze(), bins=100, weights=w
    ),
):
    """
    Solver minimizing the cost function to register an image pair.

    Parameters
    ----------
    reference_image : ndarray
        The first image of the sequence.
    moving_image : ndarray
        The second image of the sequence.
    weights : ndarray
        Weights or mask.
    channel_axis : int | None
        Index of the channel axis.
    matrix : ndarray
        Initial value of the transform, here matrix are parameters.
    model: str
        Motion model: "affine", "translation" or "euclidean".
    method: str
        Minimization method for scipy.optimization.minimize.
    options: dict
        options for scipy.optimization.minimize.
    cost: Callable, lambda (reference,moving,weights) -> float
        Cost function to be minimized taking as input the two images and
        the weights.

    Returns
    -------
    matrix : ndarray
        Homogeneous transform matrix.

    Raises
    ------
    NotImplementedError
        For unsupported motion models.

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
        matrix = np.eye(ndim + 1, dtype=float)

    parameters = _matrix_to_parameter_vector(matrix, model)

    # scale translation to image size
    scale = np.max(reference_image.shape)
    parameters = _scale_parameters(parameters, model, ndim, 1.0 / scale)

    cost = partial(
        _studholme_param_cost,
        reference_image=reference_image,
        moving_image=moving_image,
        weights=weights,
        cost=cost,
        model=model,
    )

    result = minimize(cost, x0=parameters, method=method, options=options)

    # scale translation to image size
    param = _scale_parameters(result.x, model, ndim, scale)

    return _parameter_vector_to_matrix(param, model, ndim)


def _ecc_compute_jacobian(grad, xy_grid, warp_matrix, motion_type="affine"):
    """
    Compute the Jacobian of the warp wrt the parameters.

    Parameters
    ----------
    grad : ndarray
        Gradient of the image to be warped.
    xy_grid : tuple of ndarray
        Meshgrid of the coordinates.
    warp_matrix : ndarray
        Current warping matrix.
    motion_type : {'affine', 'euclidean', 'translation'}
        Motion model 'affine', 'euclidean' or 'translation'.

    Returns
    -------
    jac : ndarray
        Jacobian of the warp wrt the parameters.
    """

    def compute_jacobian_translation(grad):
        grad_iw_x, grad_iw_y = grad
        return np.stack([grad_iw_x, grad_iw_y])

    def compute_jacobian_translation_3D(grad):
        grad_iw_x, grad_iw_y, grad_iw_z = grad
        return np.stack([grad_iw_x, grad_iw_y, grad_iw_z])

    def compute_jacobian_affine(grad, xy_grid):
        grad_iw_x, grad_iw_y = grad
        x_grid, y_grid = xy_grid

        return np.stack(
            [
                grad_iw_x * x_grid,
                grad_iw_y * x_grid,
                grad_iw_x * y_grid,
                grad_iw_y * y_grid,
                grad_iw_x,
                grad_iw_y,
            ]
        )

    def compute_jacobian_affine_3D(grad, xy_grid):
        grad_iw_x, grad_iw_y, grad_iw_z = grad
        x_grid, y_grid, z_grid = xy_grid

        return np.stack(
            [
                grad_iw_x * x_grid,
                grad_iw_y * x_grid,
                grad_iw_z * x_grid,
                grad_iw_x * y_grid,
                grad_iw_y * y_grid,
                grad_iw_z * y_grid,
                grad_iw_x * z_grid,
                grad_iw_y * z_grid,
                grad_iw_z * z_grid,
                grad_iw_x,
                grad_iw_y,
                grad_iw_z,
            ]
        )

    def compute_jacobian_euclidean(grad, xy_grid, warp_matrix):
        grad_iw_x, grad_iw_y = grad
        x_grid, y_grid = xy_grid

        h0 = warp_matrix[0, 0]
        h1 = warp_matrix[0, 1]

        hat_x = -(x_grid * h1) - (y_grid * h0)
        hat_y = (x_grid * h0) - (y_grid * h1)

        return np.stack([grad_iw_x * hat_x + grad_iw_y * hat_y, grad_iw_x, grad_iw_y])

    def compute_jacobian_homography(grad, xy_grid, warp_matrix):
        # TODO: Lets look at the paper to see if this can be made cleaner using Numpy broadcasting
        h0_ = warp_matrix[0, 0]
        h1_ = warp_matrix[1, 0]
        h2_ = warp_matrix[2, 0]
        h3_ = warp_matrix[0, 1]
        h4_ = warp_matrix[1, 1]
        h5_ = warp_matrix[2, 1]
        h6_ = warp_matrix[0, 2]
        h7_ = warp_matrix[1, 2]

        grad_iw_x, grad_iw_y = grad
        x_grid, y_grid = xy_grid

        den_ = x_grid * h2_ + y_grid * h5_ + 1.0

        grad_iw_x_ = grad_iw_x / den_
        grad_iw_y_ = grad_iw_y / den_

        hat_x = -x_grid * h0_ - y_grid * h3_ - h6_
        hat_x = np.divide(hat_x, den_)

        hat_y = -x_grid * h1_ - y_grid * h4_ - h7_
        hat_y = np.divide(hat_y, den_)

        temp = hat_x * grad_iw_x_ + hat_y * grad_iw_y_

        return np.stack(
            [
                grad_iw_x_ * x_grid,
                grad_iw_y_ * x_grid,
                temp * x_grid,
                grad_iw_x_ * y_grid,
                grad_iw_y_ * y_grid,
                temp * y_grid,
                grad_iw_x_,
                grad_iw_y_,
            ]
        )

    if np.shape(grad)[0] == 2:
        match motion_type:
            case "translation":
                return compute_jacobian_translation(grad)
            case "affine":
                return compute_jacobian_affine(grad, xy_grid)
            case "euclidean":
                return compute_jacobian_euclidean(grad, xy_grid, warp_matrix)
            case "homography":
                return compute_jacobian_homography(grad, xy_grid, warp_matrix)
    else:
        match motion_type:
            case "translation":
                return compute_jacobian_translation_3D(grad)
            case "affine":
                return compute_jacobian_affine_3D(grad, xy_grid)


def _ecc_update_warping_matrix(map_matrix, update, motion_type="affine"):
    """
    Update the warping matrix with the update vector.

    Parameters
    ----------
    map_matrix : ndarray
        Current warping matrix.
    update : ndarray
        Update vector.
    motion_type : {'affine', 'euclidean', 'translation'}
        Motion model 'affine', 'euclidean' or 'translation'.

    Returns
    -------
    map_matrix : ndarray
        Updated warping matrix.
    """

    def update_warping_matrix_translation(map_matrix, update):
        map_matrix[0, 2] += update[0]
        map_matrix[1, 2] += update[1]
        return map_matrix

    def update_warping_matrix_translation_3D(map_matrix, update):
        map_matrix[0, 3] += update[0]
        map_matrix[1, 3] += update[1]
        map_matrix[2, 3] += update[2]
        return map_matrix

    def update_warping_matrix_affine(map_matrix, update):
        map_matrix[0, 0] += update[0]
        map_matrix[1, 0] += update[1]
        map_matrix[0, 1] += update[2]
        map_matrix[1, 1] += update[3]
        map_matrix[0, 2] += update[4]
        map_matrix[1, 2] += update[5]
        return map_matrix

    def update_warping_matrix_euclidean(map_matrix, update):
        new_theta = update[0]
        new_theta += np.arcsin(map_matrix[1, 0])

        map_matrix[0, 2] += update[1]
        map_matrix[1, 2] += update[2]
        map_matrix[0, 0] = np.cos(new_theta)
        map_matrix[1, 1] = map_matrix[0, 0]
        map_matrix[1, 0] = np.sin(new_theta)
        map_matrix[0, 1] = -map_matrix[1, 0]
        return map_matrix

    def update_warping_matrix_homography(map_matrix, update):
        map_matrix[0, 0] += update[0]
        map_matrix[1, 0] += update[1]
        map_matrix[2, 0] += update[2]
        map_matrix[0, 1] += update[3]
        map_matrix[1, 1] += update[4]
        map_matrix[2, 1] += update[5]
        map_matrix[0, 2] += update[6]
        map_matrix[1, 2] += update[7]
        return map_matrix

    def update_warping_matrix_affine_3D(map_matrix, update):
        map_matrix[0, 0] += update[0]
        map_matrix[1, 0] += update[1]
        map_matrix[2, 0] += update[2]
        map_matrix[0, 1] += update[3]
        map_matrix[1, 1] += update[4]
        map_matrix[2, 1] += update[5]
        map_matrix[0, 2] += update[6]
        map_matrix[1, 2] += update[7]
        map_matrix[2, 2] += update[8]
        map_matrix[0, 3] += update[9]
        map_matrix[1, 3] += update[10]
        map_matrix[2, 3] += update[11]
        return map_matrix

    if np.shape(map_matrix)[0] == 3:
        match motion_type:
            case "translation":
                return update_warping_matrix_translation(map_matrix, update)
            case "affine":
                return update_warping_matrix_affine(map_matrix, update)
            case "euclidean":
                return update_warping_matrix_euclidean(map_matrix, update)
            case "homography":
                return update_warping_matrix_homography(map_matrix, update)
    else:
        match motion_type:
            case "translation":
                return update_warping_matrix_translation_3D(map_matrix, update)
            case "affine":
                return update_warping_matrix_affine_3D(map_matrix, update)


def _ecc_project_onto_jacobian(jac, mat):
    """
    In the orignal code the matrix is stored as a 2D [K*H,W] array, and the code is looping through K, splitting the matrix into K submatrices.Then the sub-matrix and the `mat` of size HxW are flattened into vectors.
    From there, a dot product is applied to the vectors.
    This is equivalent to multiplying the two matrices together element-by-element, then summing the result.
    Here we have it stored as a 3d array [K,H,W] `jac`, so we take advantage of broadcasting to not need to loop through K.
    """
    axis_summation = tuple(np.arange(1, len(np.shape(jac))))
    return np.sum(
        np.multiply(jac, mat), axis=axis_summation
    )  # axis=(1, 2)) if 2D, axis=(1, 2, 3)) if 3D


def _ecc_compute_hessian(jac):
    """
    the line below is equivalent to:
    hessian = np.empty((np.shape(jac)[0], np.shape(jac)[0]))
    for i in range(np.shape(jac)[0]):
        hessian[i,:] = np.sum(np.multiply(jac[i,:,:], jac), axis=(1,2))
        for j in range(i+1, np.shape(jac)[0]):
            hessian[i,j] = np.sum(np.multiply(jac[i,:,:], jac[j,:,:]))
            hessian[j,i] = hessian[i,j]
    """
    axis_summation = tuple(np.arange(1, len(np.shape(jac))))
    hessian = np.tensordot(
        jac, jac, axes=((axis_summation, axis_summation))
    )  # axes=([1, 2], [1, 2])) if 2D
    # axes=([1, 2, 3], [1, 2, 3]) if 3D
    return hessian


def custom_warp(im, mat, motion_type="affine", order=1):
    if motion_type == 'homography':

        def coord_mapping(pt, mat):
            pt += (1,)
            points_unwarping = mat @ np.array(pt).T
            return tuple(points_unwarping)

        return ndi.geometric_transform(
            im, coord_mapping, order=order, extra_arguments=(mat,)
        )
    else:
        return ndi.affine_transform(im, mat, order=order)


def solver_affine_ecc(
    reference_image,
    moving_image,
    *,
    weights,
    channel_axis,
    matrix,
    model,
    max_iter=200,
    tol=-1.0,
    order=1,
):
    """
    Estimate affine motion between two images using the Enhanced Correlation Coefficient

    Parameters
    ----------
    reference_image : ndarray
        The first image of the sequence.
    moving_image : ndarray
        The second image of the sequence.
    weights : ndarray
        Weights as an array with the same shape as reference_image.
    channel_axis : int
        Index of the channel axis.
    matrix : ndarray
        Initial homogeneous transformation matrix.
    model : {'affine', 'euclidean', 'translation'}
        Motion model 'affine', 'translation' or (2D only) 'euclidean'.
    max_iter : int
        Maximum number of inner iterations.
    tol : float
        Tolerance of the norm of the update vector.

    Returns
    -------
    matrix : ndarray
        The estimated homogeneous transformation matrix.

    Raises
    ------
    ValueError
        For numerical errors.

    Reference
    ---------
    .. [1] G. D. Evangelidis and E. Z. Psarakis, "Parametric Image Alignment Using
           Enhanced Correlation Coefficient Maximization," in IEEE Transactions on
           Pattern Analysis and Machine Intelligence, vol. 30, no. 10, pp. 1858-1865,
           Oct. 2008, doi: 10.1109/TPAMI.2008.113.
    """

    # The solver does not take into account multiple channels for now
    # Using the luminance (max across channels)
    if channel_axis is not None:
        reference_image = np.max(reference_image, channel_axis)
        moving_image = np.max(moving_image, channel_axis)

    if matrix is None:
        if len(reference_image.shape) == 2:
            matrix = np.eye(3)
        else:
            matrix = np.eye(4)

    mesh = np.meshgrid(
        *[np.arange(x, dtype=np.float32) for x in reference_image.shape], indexing='ij'
    )

    grad = np.gradient(moving_image)
    rho = -1
    last_rho = -tol

    ir_mean = np.mean(reference_image)
    ir_std = np.std(reference_image)
    ir_meancorr = reference_image - ir_mean

    ir_norm = np.sqrt(np.sum(np.prod(reference_image.shape)) * ir_std**2)

    for _ in range(max_iter):
        if np.abs(rho - last_rho) < tol:
            break

        iw_warped = custom_warp(moving_image, matrix, motion_type=model, order=order)

        iw_mean = np.mean(iw_warped[iw_warped != 0])
        iw_std = np.std(iw_warped[iw_warped != 0])
        iw_norm = np.sqrt(np.sum(iw_warped != 0) * iw_std**2)

        iw_warped_meancorr = iw_warped - iw_mean
        grad_iw_warped = np.array(
            [custom_warp(g, matrix, motion_type=model, order=order) for g in grad]
        )

        jacobian = _ecc_compute_jacobian(grad_iw_warped, mesh, matrix, model)
        hessian = _ecc_compute_hessian(jacobian)
        hessian_inv = np.linalg.inv(hessian)

        correlation = np.vdot(ir_meancorr, iw_warped_meancorr)
        last_rho = rho
        rho = correlation / (ir_norm * iw_norm)

        if np.isnan(rho):
            raise ValueError("NaN encoutered.")

        iw_projection = _ecc_project_onto_jacobian(jacobian, iw_warped_meancorr)
        ir_projection = _ecc_project_onto_jacobian(jacobian, ir_meancorr)

        iw_hessian_projection = np.matmul(hessian_inv, iw_projection)

        num = (iw_norm**2) - np.dot(iw_projection, iw_hessian_projection)
        den = correlation - np.dot(ir_projection, iw_hessian_projection)
        if den <= 0:
            warnings.warn(
                (
                    "The algorithm stopped before its convergence. The "
                    "correlation is going to be minimized. Images may "
                    "be uncorrelated or non-overlapped."
                ),
                RuntimeWarning,
            )
            return matrix

        _lambda = num / den

        error = _lambda * ir_meancorr - iw_warped_meancorr
        error_projection = _ecc_project_onto_jacobian(jacobian, error)
        delta_p = np.matmul(hessian_inv, error_projection)
        matrix = _ecc_update_warping_matrix(matrix, delta_p, model)

    return matrix


def affine(
    reference_image,
    moving_image,
    *,
    weights=None,
    channel_axis=None,
    matrix=None,
    model="affine",
    solver=solver_affine_lucas_kanade,
    pyramid_downscale=2.0,
    pyramid_minimum_size=32,
):
    """
    Coarse-to-fine affine motion estimation between two images.

    Parameters
    ----------
    reference_image : ndarray
        The first gray scale image of the sequence.
    moving_image : ndarray
        The second gray scale image of the sequence.
    weights : ndarray | None
        The weights array with same shape as reference_image.
    channel_axis : int | None
        Index of the channel axis.
    matrix : ndarray | None
        Intial guess for the homogeneous transformation matrix.
    model : {'affine', 'euclidean', 'translation'}
        Motion model: "translation", "euclidean" or "affine" where euclidean motion
        corresponds to rigid motion (rotation and translation) and affine motion can
        represent change in scale, shear, rotation and translations.
    solver: lambda(reference_image, moving_image, weights, channel_axis, matrix) -> matrix
        Affine motion solver can be solver_affine_lucas_kanade or solver_affine_studholme.
    pyramid_scale : float
        The pyramid downscale factor.
    pyramid_minimum_size : int
        The minimum size for any dimension of the pyramid levels.

    Returns
    -------
    matrix : ndarray
        The transformation matrix.

    Note
    ----
    Generate image pyramids and apply the solver at each level passing the
    previous affine parameters from the previous estimate.

    The estimated matrix can be used with func:`scipy.ndimage.affine` to register the moving image
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

    # Compute the maximum number of layers
    max_layer = int(
        floor(
            log(min(shape), pyramid_downscale)
            - log(pyramid_minimum_size, pyramid_downscale)
        )
    )

    # Generate the pyramid
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

    # Rescale the initial matrix if any to the coarsest level
    if matrix is not None:
        first_scale = pow(pyramid_downscale, -max_layer)
        parameters = _matrix_to_parameter_vector(matrix, model)
        scaled_parameters = _scale_parameters(parameters, model, ndim, first_scale)
        matrix = _parameter_vector_to_matrix(scaled_parameters, model, ndim)

    # First level
    matrix = solver(
        pyramid[0][0],
        pyramid[0][1],
        weights=pyramid[0][2],
        channel_axis=channel_axis,
        matrix=matrix,
        model=model,
    )

    # Remaining levels
    for scaled_reference_image, scaled_moving_image, scaled_weights in pyramid[1:]:
        parameters = _matrix_to_parameter_vector(matrix, model)
        parameters = _scale_parameters(parameters, model, ndim, pyramid_downscale)
        matrix = _parameter_vector_to_matrix(parameters, model, ndim)
        matrix = solver(
            reference_image=scaled_reference_image,
            moving_image=scaled_moving_image,
            weights=scaled_weights,
            channel_axis=channel_axis,
            matrix=matrix,
            model=model,
        )

    return matrix
