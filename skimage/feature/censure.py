import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter, convolve

from ..transform import integral_image
from ..feature.corner import _compute_auto_correlation
from ..morphology import convex_hull_image
from ..util import img_as_float

from .censure_cy import _censure_dob_loop


def _get_filtered_image(image, n, mode='DoB'):
    # TODO : Implement the STAR and Octagon mode
    if mode == 'DoB':
        inner_wt = (1.0 / (2 * n + 1)**2)
        outer_wt = (1.0 / (12 * n**2 + 4 * n))
        integral_img = integral_image(image)
        filtered_image = np.zeros(image.shape)
        _censure_dob_loop(image, n, integral_img, filtered_image, inner_wt, outer_wt)

        return filtered_image


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


"""
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
"""

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
    for i in xrange(no_of_scales):
        scales[:, :, i] = _get_filtered_image(image, i + 1, mode)

    # Suppressing points that are neither minima or maxima in their 3 x 3 x 3
    # neighbourhood to zero
    minimas = (minimum_filter(scales, (3, 3, 3)) == scales).astype(int) * scales
    maximas = (maximum_filter(scales, (3, 3, 3)) == scales).astype(int) * scales
    # Suppressing minimas and maximas weaker than threshold
    minimas[np.abs(minimas) < threshold] = 0
    maximas[np.abs(maximas) < threshold] = 0
    response = maximas + minimas

    for i in xrange(1, no_of_scales - 1):
        response[:, :, i] = _suppress_line(response[:, :, i], (1 + i / 3.0), rpc_threshold)

    # Returning keypoints with its scale
    keypoints = np.transpose(np.nonzero(response[:, :, 1:no_of_scales])) + [0, 0, 1]
    return keypoints
