import warnings

import numpy as np
from scipy import ndimage as ndi


def custom_warp(im, mat, motion_type="affine", order=1):
    def coord_mapping(pt, mat):
        pt += (1,)
        points_unwarping = mat @ np.array(pt).T
        return tuple(points_unwarping)

    if motion_type == 'homography':
        return ndi.geometric_transform(
            im, coord_mapping, order=order, extra_arguments=(mat,)
        )
    else:
        return ndi.affine_transform(im, mat, order=order)


def find_transform_ecc(
    ir,
    iw,
    warp_matrix=None,
    motion_type="affine",
    number_of_iterations=200,
    termination_eps=-1.0,
    gauss_filt_size=5.0,
    order=1,
):
    """
    Find the transformation matrix that aligns the warped image (iw) to the reference image (ir) using the Enhanced Correlation Coefficient (ECC) maximization.

    Parameters
    ----------
    ir : ndarray
        Reference image.
    iw : ndarray
        Warped image to be corrected.
    warp_matrix : ndarray, optional
        Initial guess for the transformation matrix, should be a 3x3 ndarray. Default is None.
    motion_type : str, optional
        Type of transformation to find. Can be "translation", "affine", "euclidean", or "homography". Default is "affine".
    number_of_iterations : int, optional
        Maximum number of iterations before the algorithm stops. Default is 200.
    termination_eps : float, optional
        Threshold for the absolute difference between the normalized correlation of successive iterations. If the difference is less than this value, the algorithm stops. Default is -1.0 (algorithm stops only after reaching the maximum number of iterations).
    gauss_filt_size : float, optional
        Standard deviation of the Gaussian kernel used for blurring ir and iw. Default is 5.0.
    order : int, optional
        Order of the interpolation used in the warp function. Default is 1.

    Returns
    -------
    warp_matrix : ndarray
        The transformation matrix that best aligns iw to ir.

    Raises
    ------
    ValueError
        If rho is NaN, indicating a failure in the algorithm.
    ValueError
        If the algorithm stops before convergence, indicating that the images may be uncorrelated or non-overlapping.
    """

    if warp_matrix is None:
        if len(ir.shape) == 2:
            warp_matrix = np.eye(3)
        else:
            warp_matrix = np.eye(4)

    mesh = np.meshgrid(*[np.arange(0, x) for x in ir.shape], indexing='ij')
    mesh = [x.astype(np.float32) for x in mesh]

    ir = ndi.gaussian_filter(ir, gauss_filt_size)
    iw = ndi.gaussian_filter(iw, gauss_filt_size)

    grad = np.gradient(iw)
    rho = -1
    last_rho = -termination_eps

    ir_mean = np.mean(ir)
    ir_std = np.std(ir)
    ir_meancorr = ir - ir_mean

    ir_norm = np.sqrt(np.sum(np.prod(ir.shape)) * ir_std**2)

    for _ in range(number_of_iterations):
        if np.abs(rho - last_rho) < termination_eps:
            break

        iw_warped = custom_warp(iw, warp_matrix, motion_type=motion_type, order=order)

        iw_mean = np.mean(iw_warped[iw_warped != 0])
        iw_std = np.std(iw_warped[iw_warped != 0])
        iw_norm = np.sqrt(np.sum(iw_warped != 0) * iw_std**2)

        iw_warped_meancorr = iw_warped - iw_mean
        grad_iw_warped = np.array(
            [
                custom_warp(g, warp_matrix, motion_type=motion_type, order=order)
                for g in grad
            ]
        )

        jacobian = compute_jacobian(grad_iw_warped, mesh, warp_matrix, motion_type)
        hessian = compute_hessian(jacobian)
        hessian_inv = np.linalg.inv(hessian)

        correlation = np.vdot(ir_meancorr, iw_warped_meancorr)
        last_rho = rho
        rho = correlation / (ir_norm * iw_norm)

        if np.isnan(rho):
            raise ValueError("NaN encoutered.")

        iw_projection = project_onto_jacobian(jacobian, iw_warped_meancorr)
        ir_projection = project_onto_jacobian(jacobian, ir_meancorr)

        iw_hessian_projection = np.matmul(hessian_inv, iw_projection)

        num = (iw_norm**2) - np.dot(iw_projection, iw_hessian_projection)
        den = correlation - np.dot(ir_projection, iw_hessian_projection)
        if den <= 0:
            warnings.warn(
                (
                    "The algorithm stopped before its convergence. The correlation is going to be minimized."
                    "Images may be uncorrelated or non-overlapped."
                ),
                RuntimeWarning,
            )
            return warp_matrix

        _lambda = num / den

        error = _lambda * ir_meancorr - iw_warped_meancorr
        error_projection = project_onto_jacobian(jacobian, error)
        delta_p = np.matmul(hessian_inv, error_projection)
        warp_matrix = update_warping_matrix(warp_matrix, delta_p, motion_type)

    return warp_matrix


def compute_jacobian(grad, xy_grid, warp_matrix, motion_type="affine"):
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


def update_warping_matrix(map_matrix, update, motion_type="affine"):
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


def project_onto_jacobian(jac, mat):
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


def compute_hessian(jac):
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
