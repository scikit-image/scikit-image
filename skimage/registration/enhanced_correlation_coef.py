import numpy as np
from scipy import ndimage as ndi

from .._shared.utils import convert_to_float
from ..transform._warps import warp

MOTION_TYPES = ["MOTION_TRANSLATION", "MOTION_EUCLIDEAN", "MOTION_AFFINE", "MOTION_HOMOGRAPHY"]


def find_transform_ECC(
    src,
    dst,
    warp_matrix,
    motion_type="MOTION_AFFINE",
    number_of_iterations=200,
    termination_eps=-1,
    input_mask=None,
    gauss_filt_size=5,
):
    src = convert_to_float(src, preserve_range=True)
    dst = convert_to_float(dst, preserve_range=True)

    if warp_matrix.shape != (3, 3):
        raise ValueError(f"warp_matrix.shape != (3,3). Current shape is: {warp_matrix.shape}")

    if motion_type not in MOTION_TYPES:
        raise ValueError(f"motion_type not in {MOTION_TYPES}")

    if warp_matrix is None:
        warp_matrix = np.eye(3)

    ws, hs = src.shape

    x_grid, y_grid = np.meshgrid(np.arange(ws), np.arange(hs))

    pre_mask = np.ones(dst.shape, dtype=np.uint8)
    if input_mask is not None:
        pre_mask = input_mask.astype(np.uint8)

    template_float = ndi.gaussian_filter(src, gauss_filt_size)

    pre_mask_float = ndi.gaussian_filter(convert_to_float(pre_mask, preserve_range=True), gauss_filt_size)

    image_float = ndi.gaussian_filter(dst, gauss_filt_size)

    gradient_y, gradient_x = np.gradient(image_float)

    gradient_x = gradient_x * pre_mask_float
    gradient_y = gradient_y * pre_mask_float

    rho = -1
    last_rho = -termination_eps

    for _ in range(number_of_iterations):
        if np.abs(rho - last_rho) < termination_eps:
            break

        image_warped = warp(image_float, warp_matrix, clip=False, order=1, preserve_range=True)
        gradient_x_warped = warp(gradient_x, warp_matrix, clip=False, order=1, preserve_range=True)
        gradient_y_warped = warp(gradient_y, warp_matrix, clip=False, order=1, preserve_range=True)
        image_mask = warp(pre_mask, warp_matrix, clip=False, order=0, preserve_range=True)

        img_mean = np.mean(image_warped[image_mask != 0])
        img_std = np.std(image_warped[image_mask != 0])

        tmp_mean = np.mean(template_float[image_mask != 0])
        tmp_std = np.std(template_float[image_mask != 0])

        image_warped[image_mask != 0] -= img_mean
        template_zm = np.zeros((hs, ws))
        template_zm[image_mask != 0] -= tmp_mean

        tmp_norm = np.sqrt(np.sum(image_mask != 0) * tmp_std**2)
        img_norm = np.sqrt(np.sum(image_mask != 0) * img_std**2)

        if motion_type == "MOTION_TRANSLATION":
            jacobian = image_jacobian_translation_ECC(gradient_x_warped, gradient_y_warped)
        if motion_type == "MOTION_AFFINE":
            jacobian = image_jacobian_affine_ECC(gradient_x_warped, gradient_y_warped, x_grid, y_grid)
        if motion_type == "MOTION_HOMOGRAPHY":
            jacobian = image_jacobian_homo_ECC(gradient_x_warped, gradient_y_warped, x_grid, y_grid, warp_matrix)
        if motion_type == "MOTION_EUCLIDEAN":
            jacobian = image_jacobian_euclidean_ECC(gradient_x_warped, gradient_y_warped, x_grid, y_grid, warp_matrix)

        hessian = project_onto_jacobian_ECC(jacobian, jacobian)
        hessian_inv = np.linalg.inv(hessian)

        correlation = np.vdot(template_zm, image_warped)

        last_rho = rho
        rho = correlation / (img_norm * tmp_norm)

        if np.isnan(rho):
            raise ValueError("NaN encountered.")

        image_projection = project_onto_jacobian_ECC(jacobian, image_warped)
        template_projection = project_onto_jacobian_ECC(jacobian, template_zm)

        image_projection_hessian = hessian_inv @ image_projection

        num = img_norm**2 - np.dot(image_projection, image_projection_hessian)
        den = correlation - np.dot(template_projection, image_projection_hessian)

        if den <= 0:
            raise ValueError(
                "den <= 0. The algorithm stopped before its convergence. Images may be uncorrelated or non-overlapped"
            )

        _lambda = num / den

        error = _lambda * template_zm - image_warped
        error_projection = project_onto_jacobian_ECC(jacobian, error)
        delta_p = hessian_inv @ error_projection

        warp_matrix = update_warping_matrix_ECC(warp_matrix, delta_p, motion_type)

    return rho, warp_matrix


def image_jacobian_translation_ECC(gradient_x_warped, gradient_y_warped):
    dst = np.empty((gradient_x_warped.shape[0], 2 * gradient_y_warped.shape[1]), dtype=np.float32)

    w = gradient_x_warped.shape[1]

    dst[:, 0:w] = gradient_x_warped
    dst[:, w : 2 * w] = gradient_y_warped

    return dst


def image_jacobian_affine_ECC(gradient_x_warped, gradient_y_warped, x_grid, y_grid):
    dst = np.empty((gradient_x_warped.shape[0], 6 * gradient_x_warped.shape[1]), dtype=np.float32)

    w = gradient_x_warped.shape[1]

    dst[:, 0:w] = np.multiply(gradient_x_warped, x_grid)
    dst[:, w : 2 * w] = np.multiply(gradient_y_warped, x_grid)
    dst[:, 2 * w : 3 * w] = np.multiply(gradient_x_warped, y_grid)
    dst[:, 3 * w : 4 * w] = np.multiply(gradient_y_warped, y_grid)
    dst[:, 4 * w : 5 * w] = gradient_x_warped
    dst[:, 5 * w : 6 * w] = gradient_y_warped

    return dst


def image_jacobian_homo_ECC(gradient_x_warped, gradient_y_warped, x_grid, y_grid, warp_matrix):
    dst = np.empty((gradient_x_warped.shape[0], 8 * gradient_x_warped.shape[1]), dtype=np.float32)

    h0 = warp_matrix[0, 0]
    h1 = warp_matrix[1, 0]
    h2 = warp_matrix[2, 0]
    h3 = warp_matrix[0, 1]
    h4 = warp_matrix[1, 1]
    h5 = warp_matrix[2, 1]
    h6 = warp_matrix[0, 2]
    h7 = warp_matrix[1, 2]

    w = gradient_x_warped.shape[1]

    # create denominator for all points as a block
    den = x_grid * h2 + y_grid * h5 + 1.0

    # create projected points
    hatX = -x_grid * h0 - y_grid * h3 - h6
    hatX = np.divide(hatX, den)

    hatY = -x_grid * h1 - y_grid * h4 - h7
    hatY = np.divide(hatY, den)

    gradient_x_warped_divided = np.divide(gradient_x_warped, den)
    gradient_y_warped_divided = np.divide(gradient_y_warped, den)

    temp = np.multiply(hatX, gradient_x_warped_divided) + np.multiply(hatY, gradient_y_warped_divided)

    # compute Jacobian blocks (8 blocks)
    dst[:, 0:w] = np.multiply(gradient_x_warped_divided, x_grid)
    dst[:, w : 2 * w] = np.multiply(gradient_y_warped_divided, x_grid)
    dst[:, 2 * w : 3 * w] = np.multiply(temp, x_grid)
    dst[:, 3 * w : 4 * w] = np.multiply(gradient_x_warped_divided, y_grid)
    dst[:, 4 * w : 5 * w] = np.multiply(gradient_y_warped_divided, y_grid)
    dst[:, 5 * w : 6 * w] = np.multiply(temp, y_grid)
    dst[:, 6 * w : 7 * w] = gradient_x_warped_divided
    dst[:, 7 * w : 8 * w] = gradient_y_warped_divided

    return dst


def image_jacobian_euclidean_ECC(gradient_x_warped, gradient_y_warped, x_grid, y_grid, warp_matrix):
    w = gradient_x_warped.shape[1]

    h0 = warp_matrix[0, 0]  # cos(theta)
    h1 = warp_matrix[1, 0]  # sin(theta)

    # create -sin(theta)*X -cos(theta)*Y for all points as a block -> hatX
    hatX = -(x_grid * h1) - (y_grid * h0)

    # create cos(theta)*X -sin(theta)*Y for all points as a block -> hatY
    hatY = (x_grid * h0) - (y_grid * h1)

    dst = np.empty((gradient_x_warped.shape[0], 3 * gradient_x_warped.shape[1]), dtype=np.float32)

    # compute Jacobian blocks (3 blocks)
    dst[:, 0:w] = np.multiply(gradient_x_warped, hatX) + np.multiply(gradient_y_warped, hatY)  # 1
    dst[:, w : 2 * w] = np.copy(gradient_x_warped)  # 2
    dst[:, 2 * w : 3 * w] = np.copy(gradient_y_warped)  # 3

    return dst


def project_onto_jacobian_ECC(src1, src2):
    if src1.shape[1] != src2.shape[1]:  # dst.cols = 1
        w = src2.shape[1]
        dst = []
        for i in range(src1.shape[1] // src2.shape[1]):
            dst.append(np.vdot(src2, src1[:, i * w : (i + 1) * w]))

        return np.array(dst)

    dst = np.empty((src1.shape[1] // src1.shape[0], src1.shape[1] // src1.shape[0]))
    w = src2.shape[1] // dst.shape[1]

    for i in range(dst.shape[1]):
        mat = src1[:, i * w : ((i + 1) * w)]
        dst[i, i] = np.linalg.norm(mat) ** 2

        for j in range(i + 1, dst.shape[1]):
            dst[j, i] = np.vdot(mat, src2[:, j * w : (j + 1) * w])
            dst[i, j] = dst[j, i]

    return dst


def update_warping_matrix_ECC(map_matrix, update, motion_type):
    if motion_type not in ["MOTION_TRANSLATION", "MOTION_EUCLIDEAN", "MOTION_AFFINE", "MOTION_HOMOGRAPHY"]:
        raise Exception(
            "motion_type not in ['MOTION_TRANSLATION','MOTION_EUCLIDEAN','MOTION_AFFINE','MOTION_HOMOGRAPHY']"
        )

    if motion_type == "MOTION_HOMOGRAPHY":
        if (map_matrix.shape[1] != 3) and (update.size != 8):
            raise Exception("(map_matrix.shape[1] != 3) and (update.size != 8)")
    elif motion_type == "MOTION_AFFINE":
        if (map_matrix.shape[1] != 2) and (update.size != 6):
            raise Exception("(map_matrix.shape[1] != 2) and (update.size != 6)")
    elif motion_type == "MOTION_EUCLIDEAN":
        if (map_matrix.shape[1] != 2) and (update.size != 3):
            raise Exception("(map_matrix.shape[1] != 2) and (update.size != 3)")
    else:
        if (map_matrix.shape[1] != 2) and (update.size != 2):
            raise Exception("(map_matrix.shape[1] != 2) and (update.size != 2)")

    if len(update.shape) != 1:
        raise Exception("len(update.shape) != 1")

    if motion_type == "MOTION_TRANSLATION":
        map_matrix[0, 2] += update[0]
        map_matrix[1, 2] += update[1]

    if motion_type == "MOTION_AFFINE":
        map_matrix[0, 0] += update[0]
        map_matrix[1, 0] += update[1]
        map_matrix[0, 1] += update[2]
        map_matrix[1, 1] += update[3]
        map_matrix[0, 2] += update[4]
        map_matrix[1, 2] += update[5]

    if motion_type == "MOTION_HOMOGRAPHY":
        map_matrix[0, 0] += update[0]
        map_matrix[1, 0] += update[1]
        map_matrix[2, 0] += update[2]
        map_matrix[0, 1] += update[3]
        map_matrix[1, 1] += update[4]
        map_matrix[2, 1] += update[5]
        map_matrix[0, 2] += update[6]
        map_matrix[1, 2] += update[7]

    if motion_type == "MOTION_EUCLIDEAN":
        new_theta = update[0]
        new_theta += np.arcsin(map_matrix[1, 0])

        map_matrix[0, 2] += update[1]
        map_matrix[1, 2] += update[2]
        map_matrix[0, 0] = map_matrix[1, 1] = np.cos(new_theta)
        map_matrix[1, 0] = np.sin(new_theta)
        map_matrix[0, 1] = -map_matrix[1, 0]
    return map_matrix
