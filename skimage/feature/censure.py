import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter

from ..transform import integral_image
from ..feature.corner import _compute_auto_correlation
from ..util import img_as_float

from .censure_cy import _censure_dob_loop, _slanted_integral_image, _censure_octagon_loop


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
        integral_img2 = np.ascontiguousarray(integral_img2)
        integral_img3 = _slanted_integral_image_modes(image, 3)
        integral_img3 = np.ascontiguousarray(integral_img3)
        integral_img4 = _slanted_integral_image_modes(image, 4)
        integral_img4 = np.ascontiguousarray(integral_img4)
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
        """
        The following figures describe area that is summed up to calculate
        the value at point @ in slanted integral image.
         _________________
        |********/        |  
        |*******/         |
        |******/          |
        |-----@           |
        |                 |
        |                 |
        |_________________|
        """
        image = np.copy(img, order='C')
        mode1 = np.zeros((image.shape[0] + 1, image.shape[1]), order='C')
        _slanted_integral_image(image, mode1)
        return mode1[1:, :]

    elif mode == 2:
        """
         _________________
        |                 |
        |                 |
        |                 |
        |           @_____|
        |          /******|
        |         /*******|
        |________/________| 
        """
        image = np.copy(img, order='C')
        image = np.fliplr(image)
        image = np.flipud(image)
        image = np.ascontiguousarray(image)
        mode2 = np.zeros((image.shape[0] + 1, image.shape[1]), order='C')
        _slanted_integral_image(image, mode2)
        mode2 = mode2[1:, :]
        mode2 = np.fliplr(mode2)
        mode2 = np.flipud(mode2)
        return mode2

    elif mode == 3:
        """
         _________________
        |                 |
        |\\               |
        |*\\              |
        |**\\             |
        |***@             |
        |***|             |
        |___|_____________| 
        """
        image = np.copy(img, order='C')
        image = np.flipud(image)
        image = image.T
        image = np.ascontiguousarray(image)
        mode3 = np.zeros((image.shape[0] + 1, image.shape[1]), order='C')
        _slanted_integral_image(image, mode3)
        mode3 = mode3[1:, :]
        mode3 = np.flipud(mode3.T)
        return mode3

    else:
        """
         ________________
        |           |****|
        |           |****|
        |           @****|
        |            \\**|
        |             \\*|
        |              \\|
        |________________|
        """
        image = np.copy(img, order='C')
        image = np.fliplr(image)
        image = image.T
        image = np.ascontiguousarray(image)
        mode4 = np.zeros((image.shape[0] + 1, image.shape[1]), order='C')
        _slanted_integral_image(image, mode4)
        mode4 = mode4[1:, :]
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
        Number of scales to extract keypoints from. The keypoints will be
        extracted from all the scales except the first and the last.

    mode : {'DoB', 'Octagon', 'STAR'}
        Type of bilevel filter used to get the scales of input image. Possible
        values are 'DoB', 'Octagon' and 'STAR'.

    threshold : float
        Threshold value used to suppress maximas and minimas with a weak
        magnitude response obtained after Non-Maximal Suppression.

    rpc_threshold : float
        Threshold for rejecting interest points which have ratio of principal
        curvatures greater than this value.

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
    scales = _get_filtered_image(image, no_of_scales, mode)

    # Suppressing points that are neither minima or maxima in their 3 x 3 x 3
    # neighbourhood to zero
    minimas = (minimum_filter(scales, (3, 3, 3)) == scales) * scales
    maximas = (maximum_filter(scales, (3, 3, 3)) == scales) * scales

    # Suppressing minimas and maximas weaker than threshold
    minimas[np.abs(minimas) < threshold] = 0
    maximas[np.abs(maximas) < threshold] = 0
    response = maximas + minimas

    for i in range(1, no_of_scales - 1):
        # sigma = (window_size - 1) / 6.0
        # window_size = 7 + 2 * i
        # Hence sigma = 1 + i / 3.0
        response[:, :, i] = _suppress_line(response[:, :, i], (1 + i / 3.0), rpc_threshold)

    # Returning keypoints with its scale
    keypoints = np.transpose(np.nonzero(response[:, :, 1:no_of_scales - 1])) + [0, 0, 2]
    return keypoints
