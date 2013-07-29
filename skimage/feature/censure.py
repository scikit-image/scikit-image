import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter

from ..transform import integral_image
from ..feature.corner import _compute_auto_correlation
from ..util import img_as_float

from .censure_cy import _censure_dob_loop, _slanted_integral_image


def _get_filtered_image(image, n, mode='DoB'):
    # TODO : Implement the STAR and Octagon mode
    if mode == 'DoB':
        inner_wt = (1.0 / (2 * n + 1)**2)
        outer_wt = (1.0 / (12 * n**2 + 4 * n))
        integral_img = integral_image(image)
        filtered_image = np.zeros(image.shape)
        _censure_dob_loop(image, n, integral_img, filtered_image, inner_wt, outer_wt)
        return filtered_image
    elif mode == 'Octagon':
        outer_shape = [(5, 2), (5, 3), (7, 3), (9, 4), (9, 7), (13, 7), (15, 10)]
        inner_shape = [(3, 0), (3, 1), (3, 2), (5, 2), (5, 3), (5, 4), (5, 5)]
        # Take these out of the loop. No need to compute again for different scales.
        integral_img = integral_image(image)
        integral_img1 = _slanted_integral_image_modes(image, 1)
        integral_img2 = _slanted_integral_image_modes(image, 2)
        integral_img3 = _slanted_integral_image_modes(image, 3)
        integral_img4 = _slanted_integral_image_modes(image, 4)
        filtered_image = np.zeros(image.shape)
        mo = outer_shape[n - 1][0]
        no = outer_shape[n - 1][1]
        mi = inner_shape[n - 1][0]
        ni = inner_shape[n - 1][1]
        outer_pixels = (mo + 2 * no)**2 - 2 * no * (no + 1)
        inner_pixels = (mi + 2 * ni)**2 - 2 * ni * (ni + 1)
        outer_wt = 1.0 / (outer_pixels - inner_pixels)
        inner_wt = 1.0 / inner_pixels
        o_m = (mo - 1) / 2
        i_m = (mi - 1) / 2
        o_set = o_m + no
        i_set = i_m + ni
        # Outsource to Cython
        for i in range(o_set + 1, image.shape[0] - o_set - 1):
            for j in range(o_set + 1, image.shape[1] - o_set - 1):
                outer = integral_img1[i + o_set, j + o_m] - integral_img1[i + o_m, j + o_set] - integral_img[i + o_set, j - o_m] + integral_img[i + o_m, j - o_m]
                outer += integral_img[i + (mo - 3) / 2, j + (mo - 3) / 2] - integral_img[i - o_m, j + (mo - 3) / 2] - integral_img[i + (mo - 3) / 2, j - o_m] + integral_img[i - o_m, j - o_m]
                outer += integral_img4[i + o_m, j - o_set] - integral_img4[i + o_set, j - o_m] - integral_img[i - o_m, j - (mo + 1) / 2] + integral_img[i - o_m, j - o_set - 1]
                outer += integral_img2[i - o_set, j - o_m] - integral_img2[i - o_m, j - o_set] - integral_img[i - (mo + 1) / 2, -1] - integral_img[i - o_set - 1, j + (mo - 3) / 2] + integral_img[i - (mo + 1) / 2, j + (mo - 3) / 2] + integral_img[i - o_set - 1, -1]
                outer += integral_img3[i - o_m, j + o_set] - integral_img3[i - o_set, j + o_m] - integral_img[-1, j + o_set + 1] - integral_img[i + (mo - 3) / 2, j + o_m] + integral_img[-1, j + o_m] + integral_img[i + (mo - 3) / 2, j + o_set + 1]

                inner = integral_img1[i + i_set, j + i_m] - integral_img1[i + i_m, j + i_set] - integral_img[i + i_set, j - i_m] + integral_img[i + i_m, j - i_m]
                inner += integral_img[i + (mi - 3) / 2, j + (mi - 3) / 2] - integral_img[i - i_m, j + (mi - 3) / 2] - integral_img[i + (mi - 3) / 2, j - i_m] + integral_img[i - i_m, j - i_m]
                inner += integral_img4[i + i_m, j - i_set] - integral_img4[i + i_set, j - i_m] - integral_img[i - i_m, j - (mi + 1) / 2] + integral_img[i - i_m, j - i_set - 1]
                inner += integral_img2[i - i_set, j - i_m] - integral_img2[i - i_m, j - i_set] - integral_img[i - (mi + 1) / 2, -1] - integral_img[i - i_set - 1, j + (mi - 3) / 2] + integral_img[i - (mi + 1) / 2, j + (mi - 3) / 2] + integral_img[i - i_set - 1, -1]
                inner += integral_img3[i - i_m, j + i_set] - integral_img3[i - i_set, j + i_m] - integral_img[-1, j + i_set + 1] - integral_img[i + (mi - 3) / 2, j + i_m] + integral_img[-1, j + i_m] + integral_img[i + (mi - 3) / 2, j + i_set + 1]

                filtered_image[i, j] = outer_wt * outer - (outer_wt + inner_wt) * inner
        return filtered_image


# Outsource to Cython
def _slanted_integral_image(image):
    flipped_lr = np.fliplr(image)
    left_sum = np.zeros(image.shape[0])
    for i in range(image.shape[1] - image.shape[0], image.shape[1]):
        left_sum[image.shape[1] - 1 - i] = np.sum(flipped_lr.diagonal(i))
    left_sum = left_sum.cumsum(0)
    right_sum = np.sum(image, 1).cumsum(0)
    image[:, 0] = left_sum
    image[:, -1] = right_sum
    integral_img = np.zeros((image.shape[0] + 1, image.shape[1]))
    integral_img[1:, :] = image
    for i in range(1, integral_img.shape[0]):
        for j in range(1, integral_img.shape[1] - 1):
            integral_img[i, j] += integral_img[i, j - 1] + integral_img[i - 1, j + 1] - integral_img[i - 1, j]
    return integral_img[1:, :integral_img.shape[1]]


def _slanted_integral_image_modes(img, mode=1):
    if mode == 1:
        image = np.copy(img)
        mode1 = _slanted_integral_image(image, 1)
        return mode1

    elif mode == 2:
        image = np.copy(img)
        image = np.fliplr(image)
        image = np.flipud(image)
        mode2 = _slanted_integral_image(image, 2)
        mode2 = np.fliplr(mode2)
        mode2 = np.flipud(mode2)
        return mode2

    elif mode == 3:
        image = np.copy(img)
        image = np.flipud(image)
        image = image.T
        mode3 = _slanted_integral_image(image, 3)
        mode3 = np.flipud(mode3.T)
        return mode3

    else:
        image = np.copy(img)
        image = np.fliplr(image)
        image = image.T
        mode4 = _slanted_integral_image(image, 4)
        mode4 = np.fliplr(mode4.T)
        return mode4


def _suppress_line(response, sigma, rpc_threshold):
    Axx, Axy, Ayy = _compute_auto_correlation(response, sigma)
    detA = Axx * Ayy - Axy**2
    traceA = Axx + Ayy
    # ratio of principal curvatures
    rpc = traceA**2 / (detA + 0.001)
    response[rpc > rpc_threshold] = 0
    return response


def censure_keypoints(image, mode='DoB', no_of_scales=7, threshold=0.03, rpc_threshold=10):
    # TODO : Decide number of scales. Image-size dependent?
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")
    image = img_as_float(image)

    image = np.ascontiguousarray(image)

    # Generating all the scales
    scales = np.zeros((image.shape[0], image.shape[1], no_of_scales))
    for i in range(no_of_scales):
        scales[:, :, i] = _get_filtered_image(image, i + 1, mode)

    # Suppressing points that are neither minima or maxima in their 3 x 3 x 3
    # neighbourhood to zero
    minimas = (minimum_filter(scales, (3, 3, 3)) == scales).astype(int) * scales
    maximas = (maximum_filter(scales, (3, 3, 3)) == scales).astype(int) * scales
    # Suppressing minimas and maximas weaker than threshold
    minimas[np.abs(minimas) < threshold] = 0
    maximas[np.abs(maximas) < threshold] = 0
    response = maximas + minimas

    for i in range(1, no_of_scales - 1):
        response[:, :, i] = _suppress_line(response[:, :, i], (1 + i / 3.0), rpc_threshold)

    # Returning keypoints with its scale
    keypoints = np.transpose(np.nonzero(response[:, :, 1:no_of_scales])) + [0, 0, 1]
    return keypoints
