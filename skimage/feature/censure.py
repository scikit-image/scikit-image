import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter

from ..transform import integral_image
from ..feature.corner import _compute_auto_correlation
from ..util import img_as_float

from .censure_cy import _censure_dob_loop, _slanted_integral_image, _censure_octagon_loop
from time import time

def _get_filtered_image(image, no_of_scales, mode):
    # TODO : Implement the STAR mode
    if mode == 'DoB':
        scales = np.zeros((image.shape[0], image.shape[1], no_of_scales))
        for i in range(no_of_scales):
            n = i + 1
            inner_wt = (1.0 / (2 * n + 1)**2)
            outer_wt = (1.0 / (12 * n**2 + 4 * n))
            integral_img = integral_image(image)
            filtered_image = np.zeros(image.shape)
            _censure_dob_loop(image, n, integral_img, filtered_image, inner_wt, outer_wt)
            scales[:, :, i] = filtered_image
        return scales
    elif mode == 'Octagon':
        # TODO : Decide the shapes of Octagon filters for scales > 7
        outer_shape = [(5, 2), (5, 3), (7, 3), (9, 4), (9, 7), (13, 7), (15, 10)]
        inner_shape = [(3, 0), (3, 1), (3, 2), (5, 2), (5, 3), (5, 4), (5, 5)]
        scales = np.zeros((image.shape[0], image.shape[1], no_of_scales))
        integral_img = integral_image(image)
        integral_img1 = _slanted_integral_image_modes(image, 1)
        integral_img2 = _slanted_integral_image_modes(image, 2)
        integral_img3 = _slanted_integral_image_modes(image, 3)
        integral_img4 = _slanted_integral_image_modes(image, 4)
        for k in range(no_of_scales):
            n = k + 1
            filtered_image = np.zeros(image.shape)
            mo = outer_shape[n - 1][0]
            no = outer_shape[n - 1][1]
            mi = inner_shape[n - 1][0]
            ni = inner_shape[n - 1][1]
            outer_pixels = (mo + 2 * no)**2 - 2 * no * (no + 1)
            inner_pixels = (mi + 2 * ni)**2 - 2 * ni * (ni + 1)
            outer_wt = 1.0 / (outer_pixels - inner_pixels)
            inner_wt = 1.0 / inner_pixels

            _censure_octagon_loop(image, integral_img, integral_img1, integral_img2, integral_img3, integral_img4, filtered_image, outer_wt, inner_wt, mo, no, mi, ni)

            scales[:, :, k] = filtered_image
        return scales


def _slanted_integral_image_modes(img, mode=1):
    if mode == 1:
        image = np.copy(img)
        mode1 = _slanted_integral_image(image)
        return mode1

    elif mode == 2:
        image = np.copy(img)
        image = np.fliplr(image)
        image = np.flipud(image)
        mode2 = _slanted_integral_image(image)
        mode2 = np.fliplr(mode2)
        mode2 = np.flipud(mode2)
        return mode2

    elif mode == 3:
        image = np.copy(img)
        image = np.flipud(image)
        image = image.T
        mode3 = _slanted_integral_image(image)
        mode3 = np.flipud(mode3.T)
        return mode3

    else:
        image = np.copy(img)
        image = np.fliplr(image)
        image = image.T
        mode4 = _slanted_integral_image(image)
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


def censure_keypoints(image, no_of_scales=7, mode='DoB', threshold=0.03, rpc_threshold=10):
    """
    Extracts Censure keypoints along with the corresponding scale using
    either Difference of Boxes, Octagon or STAR bilevel filter.

    Parameters
    ----------
    image : 2D ndarray
        Input image.

    no_of_scales : positive integer
        Number of scales to extract keypoints from. Default is 7.

    mode : 'DoB'
        Type of bilevel filter used to get the scales of input image. Possible
        values are 'DoB', 'Octagon' and 'STAR'. Default is 'DoB'.

    threshold :
        Threshold value used to suppress maximas and minimas with a weak
        magnitude response obtained after Non-Maximal Suppression. Default
        is 0.03.

    rpc_threshold :
        Threshold for rejecting interest points which have ratio of principal
        curvatures greater than this value. Default is 10.

    Returns
    -------
    keypoints : (N, 3) array
        Location of extracted keypoints along with the corresponding scale.

    References
    ----------
    .. [1] Motilal Agrawal, Kurt Konolige and Morten Rufus Blas
           "CenSurE: Center Surround Extremas for Realtime Feature
           Detection and Matching",
           http://link.springer.com/content/pdf/10.1007%2F978-3-540-88693-8_8.pdf

    .. [2] Adam Schmidt, Marek Kraft, Michal Fularz and Zuzanna Domagala
           "Comparative Assessment of Point Feature Detectors and
           Descriptors in the Context of Robot Navigation" 
           http://www.jamris.org/01_2013/saveas.php?QUEST=JAMRIS_No01_2013_P_11-20.pdf

    """

    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")
    image = img_as_float(image)

    image = np.ascontiguousarray(image)

    # Generating all the scales
    start = time()
    scales = np.zeros((image.shape[0], image.shape[1], no_of_scales))
    scales = _get_filtered_image(image, no_of_scales, mode)
    print time() - start

    # Suppressing points that are neither minima or maxima in their 3 x 3 x 3
    # neighbourhood to zero
    minimas = (minimum_filter(scales, (3, 3, 3)) == scales).astype(int) * scales
    maximas = (maximum_filter(scales, (3, 3, 3)) == scales).astype(int) * scales
    print time() - start
    # Suppressing minimas and maximas weaker than threshold
    minimas[np.abs(minimas) < threshold] = 0
    maximas[np.abs(maximas) < threshold] = 0
    response = maximas + minimas
    print time() - start
    for i in range(1, no_of_scales - 1):
        response[:, :, i] = _suppress_line(response[:, :, i], (1 + i / 3.0), rpc_threshold)
    print time() - start
    # Returning keypoints with its scale
    keypoints = np.transpose(np.nonzero(response[:, :, 1:no_of_scales])) + [0, 0, 1]
    return keypoints
