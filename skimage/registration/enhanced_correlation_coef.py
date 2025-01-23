import numpy as np
from scipy import ndimage as ndi

from ..transform._warps import warp


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
    """find_transform_ecc _summary_

    Parameters
    ----------
    ir : ndarray
        reference image
    iw : ndarray
        warped image to be corrected
    warp_matrix : ndarray, optional
        initial guess for the transformation matrix, should be a 3x3 ndarray by default None
    motion_type : str, optional
        determine the type of transformation the algorithm will try to find. Can be either "translation", "affine", "euclidean" or "homography", by default "affine"
    number_of_iterations : int, optional
        number of iterations before the algorithm stops (will stop earlier if it reaches termination_eps), by default 200
    termination_eps : float, optional
        If the absolute difference between the normalized correlation from two successive run is less than this, the algorithm will stop, by default -1.0 (Which means it will stop only when it reaches number_of_iterations)
    gauss_filt_size : float, optional
        Standard deviation of the gaussian kernel used for bluring ir and iw, by default 5
    order : int, optional
        Order of the interpolation, see 'warp' for more details, by default 1

    Returns
    -------
    warp_matrix: ndarray
        The matrix that will warp iw to ir.

    Raises
    ------
    ValueError
        if rho is a NaN the algorithm will stop
    ValueError
        _description_
    """
    if warp_matrix is None:
        warp_matrix = np.eye(3)

    x_grid, y_grid = np.meshgrid(np.arange(ir.shape[0]), np.arange(iw.shape[1]))
    x_grid = x_grid.astype(np.float32)
    y_grid = y_grid.astype(np.float32)
    ir = ndi.gaussian_filter(ir, gauss_filt_size)
    iw = ndi.gaussian_filter(iw, gauss_filt_size)

    [grad_iw_y, grad_iw_x] = np.gradient(iw)

    rho = -1
    last_rho = -termination_eps

    ir_mean = np.mean(ir)
    ir_std = np.std(ir)
    ir_meancorr = ir - ir_mean

    ir_norm = np.sqrt(np.sum(np.prod(ir.shape)) * ir_std**2)

    for _ in range(number_of_iterations):
        if np.abs(rho - last_rho) < termination_eps:
            break

        iw_warped = warp(iw, warp_matrix, order=order)

        # TODO: This need to be corrected to not take into account the out-of-bound value (set to 0)
        iw_mean = np.mean(iw_warped[iw_warped != 0])
        iw_std = np.std(iw_warped[iw_warped != 0])
        iw_norm = np.sqrt(np.sum(iw_warped != 0) * iw_std**2)
        # iw_norm = np.sqrt(np.sum(np.prod(iw.shape)) * iw_std**2)

        iw_warped_meancorr = iw_warped - iw_mean

        grad_iw_x_warped = warp(grad_iw_x, warp_matrix, order=order)
        grad_iw_y_warped = warp(grad_iw_y, warp_matrix, order=order)

        jacobian = compute_jacobian(
            [grad_iw_x_warped, grad_iw_y_warped],
            [x_grid, y_grid],
            warp_matrix,
            motion_type,
        )
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
            raise ValueError(
                "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped."
            )

        _lambda = num / den

        error = _lambda * ir_meancorr - iw_warped_meancorr
        error_projection = project_onto_jacobian(jacobian, error)
        delta_p = np.matvec(hessian_inv, error_projection)
        warp_matrix = update_warping_matrix(warp_matrix, delta_p, motion_type)

    return warp_matrix


def compute_jacobian(grad, xy_grid, warp_matrix, motion_type="affine"):
    def compute_jacobian_translation(grad):
        grad_iw_x, grad_iw_y = grad
        return np.stack([grad_iw_x, grad_iw_y])

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

    match motion_type:
        case "translation":
            return compute_jacobian_translation(grad)
        case "affine":
            return compute_jacobian_affine(grad, xy_grid)
        case "euclidean":
            return compute_jacobian_euclidean(grad, xy_grid, warp_matrix)
        case "homography":
            return compute_jacobian_homography(grad, xy_grid, warp_matrix)


def update_warping_matrix(map_matrix, update, motion_type="affine"):
    def update_warping_matrix_translation(map_matrix, update):
        map_matrix[0, 2] += update[0]
        map_matrix[1, 2] += update[1]
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

    match motion_type:
        case "translation":
            return update_warping_matrix_translation(map_matrix, update)
        case "affine":
            return update_warping_matrix_affine(map_matrix, update)
        case "euclidean":
            return update_warping_matrix_euclidean(map_matrix, update)
        case "homography":
            return update_warping_matrix_homography(map_matrix, update)


def project_onto_jacobian(jac, mat):
    """
    In the orignal code the matrix is stored as a 2D [K*H,W] array, and the code is looping through K, splitting the matrix into K submatrices.Then the sub-matrix and the `mat` of size HxW are flattened into vectors.
    From there, a dot product is applied to the vectors.
    This is equivalent to multiplying the two matrices together element-by-element, then summing the result.
    Here we have it stored as a 3d array [K,H,W] `jac`, so we take advantage of broadcasting to not need to loop through K.
    """
    return np.sum(np.multiply(jac, mat), axis=(1, 2))


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
    hessian = np.tensordot(jac, jac, axes=([1, 2], [1, 2]))
    return hessian
