import numpy as np
from scipy import ndimage as ndi

from .._shared.utils import convert_to_float
from ..transform._warps import warp

MOTION_TYPES = ["MOTION_TRANSLATION", "MOTION_EUCLIDEAN", "MOTION_AFFINE", "MOTION_HOMOGRAPHY"]


def find_transform_ECC(
    src,
    dst,
    warp_matrix=None,
    motion_type="MOTION_AFFINE",
    number_of_iterations=200,
    termination_eps=-1,
    input_mask=None,
    gauss_filt_size=5,
    numerical_dtype=np.float32,
):
    if warp_matrix is None:
        warp_matrix = np.eye(3, dtype=numerical_dtype)

    if src.dtype != dst.dtype:
        raise TypeError(
            "Both input images must have the same data type. \n Current dtype is: template_image-{src.dtype}, input_image-{dst.dtype}"
        )

    if warp_matrix.shape != (3, 3) and warp_matrix.shape[1] != 2:
        raise ValueError(f"warp_matrix.shape != (3,3). Current shape is: {warp_matrix.shape}")

    if motion_type not in MOTION_TYPES:
        raise ValueError(f"motion_type not in {MOTION_TYPES}")

    hs, ws = src.shape
    wd, hd = dst.shape

    x_grid, y_grid = np.meshgrid(np.arange(ws), np.arange(hs))
    x_grid = x_grid.astype(numerical_dtype)
    y_grid = y_grid.astype(numerical_dtype)

    # to use it for mask warping
    pre_mask = np.ones((hd, wd), dtype=np.uint8)
    if input_mask is not None:
        pre_mask = input_mask

    # gaussian filtering is optional
    template_float = ndi.gaussian_filter(src, gauss_filt_size)
    template_float = template_float.astype(numerical_dtype)

    pre_maskFloat = pre_mask.astype(numerical_dtype)
    pre_maskFloat = ndi.gaussian_filter(pre_maskFloat, gauss_filt_size)
    pre_maskFloat = pre_maskFloat.astype(numerical_dtype)
    # Change threshold.
    pre_maskFloat = pre_maskFloat * (0.5 / 0.95)

    # Rounding conversion.
    pre_mask = np.round(pre_maskFloat).astype(pre_mask.dtype)
    pre_maskFloat = pre_mask.astype(pre_maskFloat.dtype)

    imageFloat = dst.astype(numerical_dtype)
    imageFloat = ndi.gaussian_filter(imageFloat, gauss_filt_size)

    # calculate first order image derivatives
    [gradientY, gradientX] = np.gradient(imageFloat)

    gradientX = np.multiply(gradientX, pre_maskFloat)
    gradientY = np.multiply(gradientY, pre_maskFloat)
    gradientX = gradientX.astype(numerical_dtype)
    gradientY = gradientY.astype(numerical_dtype)

    # iteratively update map_matrix
    rho = -1
    last_rho = -termination_eps
    for _ in range(number_of_iterations):
        if np.abs(rho - last_rho) < termination_eps:
            break

        imageWarped = warp(imageFloat, warp_matrix, clip=False, order=1, preserve_range=True)
        gradient_x_warped = warp(gradientX, warp_matrix, clip=False, order=1, preserve_range=True)
        gradient_y_warped = warp(gradientY, warp_matrix, clip=False, order=1, preserve_range=True)
        imageMask = warp(pre_mask, warp_matrix, clip=False, order=0, preserve_range=True)

        imgMean = np.mean(imageWarped[imageMask != 0])
        imgStd = np.std(imageWarped[imageMask != 0])
        tmpMean = np.mean(template_float[imageMask != 0])
        tmpStd = np.std(template_float[imageMask != 0])

        imageWarped[imageMask != 0] = imageWarped[imageMask != 0] - imgMean
        templateZM = np.zeros((hs, ws), dtype=np.float32)
        templateZM[imageMask != 0] = template_float[imageMask != 0] - tmpMean

        tmpNorm = np.sqrt(np.sum(imageMask != 0) * (tmpStd**2))
        imgNorm = np.sqrt(np.sum(imageMask != 0) * (imgStd**2))

        # calculate jacobian of image wrt parameters
        if motion_type == "MOTION_AFFINE":
            jacobian = image_jacobian_affine_ECC(gradient_x_warped, gradient_y_warped, x_grid, y_grid)
        if motion_type == "MOTION_HOMOGRAPHY":
            jacobian = image_jacobian_homo_ECC(gradient_x_warped, gradient_y_warped, x_grid, y_grid, warp_matrix)
        if motion_type == "MOTION_TRANSLATION":
            jacobian = image_jacobian_translation_ECC(gradient_x_warped, gradient_y_warped)
        if motion_type == "MOTION_EUCLIDEAN":
            jacobian = image_jacobian_euclidean_ECC(gradient_x_warped, gradient_y_warped, x_grid, y_grid, warp_matrix)

        hessian = project_onto_jacobian_ECC(jacobian, jacobian)
        hessian = hessian.astype(numerical_dtype)
        hessianInv = np.linalg.inv(hessian)

        correlation = np.vdot(templateZM, imageWarped)

        last_rho = rho
        rho = correlation / (imgNorm * tmpNorm)

        if np.isnan(rho):
            raise Exception("NaN encountered.")

        # project images into jacobian
        imageProjection = project_onto_jacobian_ECC(jacobian, imageWarped)
        templateProjection = project_onto_jacobian_ECC(jacobian, templateZM)

        # calculate the parameter lambda to account for illumination variation
        imageProjectionHessian = np.matmul(hessianInv, imageProjection)
        num = (imgNorm * imgNorm) - np.dot(imageProjection, imageProjectionHessian)
        den = correlation - np.dot(templateProjection, imageProjectionHessian)

        if den <= 0.0:
            rho = -1
            raise Exception(
                "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped"
            )

        _lambda = num / den

        # estimate the update step delta_p
        error = _lambda * templateZM - imageWarped
        errorProjection = project_onto_jacobian_ECC(jacobian, error)
        deltaP = np.matmul(hessianInv, errorProjection)

        # update warping matrix
        warp_matrix = update_warping_matrix_ECC(warp_matrix, deltaP, motion_type)

    return rho, warp_matrix


def image_jacobian_translation_ECC(gradient_x_warped, gradient_y_warped):
    if gradient_x_warped.size != gradient_y_warped.size:
        raise Exception("gradient_x_warped.size != gradient_y_warped.size")

    dst = np.empty((gradient_x_warped.shape[0], 2 * gradient_x_warped.shape[1]), dtype=np.float32)

    w = gradient_x_warped.shape[1]

    # compute Jacobian blocks (2 blocks)

    dst[:, 0:w] = np.copy(gradient_x_warped)
    dst[:, w : 2 * w] = np.copy(gradient_y_warped)

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
    h0_ = warp_matrix[0, 0]
    h1_ = warp_matrix[1, 0]
    h2_ = warp_matrix[2, 0]
    h3_ = warp_matrix[0, 1]
    h4_ = warp_matrix[1, 1]
    h5_ = warp_matrix[2, 1]
    h6_ = warp_matrix[0, 2]
    h7_ = warp_matrix[1, 2]

    w = gradient_x_warped.shape[1]

    # create denominator for all points as a block
    den_ = x_grid * h2_ + y_grid * h5_ + 1.0  # check the time of this! otherwise use addWeighted

    # create projected points
    hat_x = -x_grid * h0_ - y_grid * h3_ - h6_
    hat_x = np.divide(hat_x, den_)

    hat_y = -x_grid * h1_ - y_grid * h4_ - h7_
    hat_y = np.divide(hat_y, den_)

    # instead of dividing each block with den,
    # just pre-divide the block of gradients (it's more efficient)

    gradient_x_warped_divided = np.divide(gradient_x_warped, den_)
    gradient_y_warped_divided = np.divide(gradient_y_warped, den_)

    temp = np.multiply(hat_x, gradient_x_warped_divided) + np.multiply(hat_y, gradient_y_warped_divided)

    dst = np.empty((gradient_x_warped.shape[0], 8 * gradient_x_warped.shape[1]), dtype=np.float32)

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
    dst[:, 0:w] = np.multiply(gradient_x_warped, hatX) + np.multiply(gradient_y_warped, hatY)
    dst[:, w : 2 * w] = gradient_x_warped
    dst[:, 2 * w : 3 * w] = gradient_y_warped

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
    if motion_type not in MOTION_TYPES:
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
