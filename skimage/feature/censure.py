import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter, convolve
from skimage.transform import integral_image
from skimage.feature.corner import _compute_auto_correlation
from skimage.morphology import convex_hull_image
import time


"""
def _get_filtered_image(image, n, mode='DoB'):
    # TODO : Implement the STAR and Octagon mode
    if mode == 'DoB':
        inner_wt = (1.0 / (2*n + 1)**2)
        outer_wt = (1.0 / (12*n**2 + 4*n))
        integral_img = integral_image(image)
        filtered_image = np.zeros(image.shape)
        # TODO : Outsource to Cython
        start = time.time()
        for i in range(2 * n, image.shape[0] - 2 * n):
            for j in range(2 * n, image.shape[1] - 2 * n):
                inner = integral_img[i + n, j + n] + integral_img[i - n - 1, j - n - 1] - integral_img[i + n, j - n - 1] - integral_img[i - n - 1, j + n]
                outer = integral_img[i + 2 * n, j + 2 * n] + integral_img[i - 2 * n - 1, j - 2 * n - 1] - integral_img[i + 2 * n, j - 2 * n - 1] - integral_img[i - 2 * n - 1, j + 2 * n]
                filtered_image[i, j] = outer_wt * outer - (inner_wt + outer_wt) * inner
        print time.time() - start
        return filtered_image
    elif mode == 'Octagon':
        outer_shape = [(5, 2), (5, 3), (7, 3), (9, 4), (9, 7), (13, 7), (15, 10)]
        inner_shape = [(3, 0), (3, 1), (3, 2), (5, 2), (5, 3), (5, 4), (5, 5)]
"""


def _oct(m, n):
    f = np.zeros((m + 2*n, m + 2*n))
    f[0, n] = 1
    f[n, 0] = 1
    f[0, m + n -1] = 1
    f[m + n - 1, 0] = 1
    f[-1, n] = 1 
    f[n, -1] = 1
    f[-1, m + n - 1] = 1
    f[m + n - 1, -1] = 1
    return convex_hull_image(f).astype(int)


def _octagon_filter(mo, no, mi, ni):
    outer = (mo + 2 * no)**2 - 2 * no * (no + 1)
    inner = (mi + 2 * ni)**2 - 2 * ni * (ni + 1)
    outer_wt = 1.0 / (outer - inner)
    inner_wt = 1.0 / inner
    c = ((mo + 2 * no) - (mi + 2 * ni)) / 2
    outer_oct = _oct(mo, no)
    inner_oct = np.zeros((mo + 2 * no, mo + 2 * no))
    inner_oct[c:-c, c:-c] = _oct(mi, ni)
    bfilter = outer_wt * outer_oct - (outer_wt + inner_wt) * inner_oct
    return bfilter


def _filter_using_convolve(image, n, mode='DoB'):

    if mode == 'DoB':
        inner_wt = (1.0 / (2*n + 1)**2)
        outer_wt = (1.0 / (12*n**2 + 4*n))
        dob_filter = np.zeros((4 * n + 1, 4 * n + 1))
        dob_filter[:] = outer_wt
        dob_filter[n : 3 * n + 1, n : 3 * n + 1] = - inner_wt
        return convolve(image, dob_filter)

    elif mode == 'Octagon':
        outer_shape = [(5, 2), (5, 3), (7, 3), (9, 4), (9, 7), (13, 7), (15, 10)]
        inner_shape = [(3, 0), (3, 1), (3, 2), (5, 2), (5, 3), (5, 4), (5, 5)]
        return convolve(image, _octagon_filter(outer_shape[n - 1][0], outer_shape[n - 1][1], inner_shape[n - 1][0], inner_shape[n - 1][1]))


def _suppress_line(response, sigma, rpc_threshold):
    Axx, Axy, Ayy = _compute_auto_correlation(response, sigma)
    detA = Axx * Ayy - Axy**2
    traceA = Axx + Ayy
    # ratio of principal curvatures
    rpc = traceA / (detA + 0.001)
    response[rpc > rpc_threshold] = 0
    return response


def censure_keypoints(image, mode='DoB', threshold=0.03, rpc_threshold=10):
    # TODO : Decide number of scales. Image-size dependent?
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")
    # Generating all the scales
    start = time.time()
    scale1 = _filter_using_convolve(image, 1, mode)
    scale2 = _filter_using_convolve(image, 2, mode)
    scale3 = _filter_using_convolve(image, 3, mode)
    scale4 = _filter_using_convolve(image, 4, mode)
    scale5 = _filter_using_convolve(image, 5, mode)
    scale6 = _filter_using_convolve(image, 6, mode)
    scale7 = _filter_using_convolve(image, 7, mode)
    print time.time() - start
    # Stacking all the scales in the 3rd dimension
    scales = np.dstack((scale1, scale2, scale3, scale4, scale5, scale6, scale7))
    # Suppressing points that are neither minima or maxima in their 3 x 3 x 3
    # neighbourhood to zero
    minimas = (minimum_filter(scales, (3, 3, 3)) == scales).astype(int) * scales
    maximas = (maximum_filter(scales, (3, 3, 3)) == scales).astype(int) * scales
    # Suppressing minimas and maximas weaker than threshold
    minimas[np.abs(minimas) < threshold] = 0
    maximas[np.abs(maximas) < threshold] = 0
    response = maximas + np.abs(minimas)
    # TODO : Decide the rpc_threshold and sigma for all the scales. The paper only discusses
    # values for scale2 i.e. response[:, :, 1]
    response[:, :, 1] = _suppress_line(response[:, :, 1], 1.33, rpc_threshold)
    response[:, :, 2] = _suppress_line(response[:, :, 2], 1.33, rpc_threshold)
    response[:, :, 3] = _suppress_line(response[:, :, 3], 1.33, rpc_threshold)
    response[:, :, 4] = _suppress_line(response[:, :, 4], 1.33, rpc_threshold)
    response[:, :, 5] = _suppress_line(response[:, :, 5], 1.33, rpc_threshold)
    # TODO : Return key-points from all the scales?
    return response
