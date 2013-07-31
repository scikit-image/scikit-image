import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter, convolve

from ..transform import integral_image
from ..feature.corner import _compute_auto_correlation
from ..util import img_as_float
from ..morphology import convex_hull_image

from .censure_cy import _censure_dob_loop


def _get_filtered_image(image, n_scales, mode):
    # TODO : Implement the STAR mode
    scales = np.zeros((image.shape[0], image.shape[1], n_scales),
                      dtype=np.double)

    if mode == 'DoB':
        for i in range(n_scales):
            n = i + 1

            # Constant multipliers for the outer region and the inner region
            # of the bilevel filters with the constraint of keeping the
            # DC bias 0.
            inner_weight = (1.0 / (2 * n + 1)**2)
            outer_weight = (1.0 / (12 * n**2 + 4 * n))

            integral_img = integral_image(image)

            filtered_image = np.zeros(image.shape)
            _censure_dob_loop(image, n, integral_img, filtered_image,
                              inner_weight, outer_weight)

            scales[:, :, i] = filtered_image

    # NOTE : For the Octagon shaped filter, we implemented and evaluated the
    # slanted integral image based image filtering but the performance was
    # more or less equal to image filtering using
    # scipy.ndimage.filters.convolve(). Hence we have decided to use the
    # later for a much cleaner implementation.
    elif mode == 'Octagon':
        # TODO : Decide the shapes of Octagon filters for scales > 7
        outer_shape = [(5, 2), (5, 3), (7, 3), (9, 4), (9, 7), (13, 7),
                       (15, 10)]
        inner_shape = [(3, 0), (3, 1), (3, 2), (5, 2), (5, 3), (5, 4), (5, 5)]

        # 
        for i in range(n_scales):
            scales[:, :, i] = convolve(image,
                                       _octagon_filter(outer_shape[i][0],
                                       outer_shape[i][1], inner_shape[i][0],
                                       inner_shape[i][1]))

    return scales


# TODO : Import from selem after getting #669 merged.
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


def _suppress_line(response, sigma, rpc_threshold):
    Axx, Axy, Ayy = _compute_auto_correlation(response, sigma)
    detA = Axx * Ayy - Axy**2
    traceA = Axx + Ayy

    # ratio of principal curvatures
    rpc = traceA**2 / (detA + 0.001)
    response[rpc > rpc_threshold] = 0
    return response


def censure_keypoints(image, n_scales=7, mode='DoB', threshold=0.03,
                      rpc_threshold=10):
    """
    Extracts Censure keypoints along with the corresponding scale using
    either Difference of Boxes, Octagon or STAR bilevel filter.

    Parameters
    ----------
    image : 2D ndarray
        Input image.

    n_scales : positive integer
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
    scales = _get_filtered_image(image, n_scales, mode)

    # Suppressing points that are neither minima or maxima in their 3 x 3 x 3
    # neighbourhood to zero
    minimas = (minimum_filter(scales, (3, 3, 3)) == scales) * scales
    maximas = (maximum_filter(scales, (3, 3, 3)) == scales) * scales

    # Suppressing minimas and maximas weaker than threshold
    minimas[np.abs(minimas) < threshold] = 0
    maximas[np.abs(maximas) < threshold] = 0
    response = maximas + minimas

    for i in range(1, n_scales - 1):
        # sigma = (window_size - 1) / 6.0
        # window_size = 7 + 2 * i
        # Hence sigma = 1 + i / 3.0
        response[:, :, i] = _suppress_line(response[:, :, i], (1 + i / 3.0),
                                           rpc_threshold)

    # Returning keypoints with its scale
    keypoints = (np.transpose(np.nonzero(response[:, :, 1:n_scales - 1]))
                 + [0, 0, 2])
    return keypoints
