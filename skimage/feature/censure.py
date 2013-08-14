import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter, convolve

from skimage.transform import integral_image
from skimage.feature.corner import _compute_auto_correlation
from skimage.util import img_as_float
from skimage.morphology import convex_hull_image
from skimage.feature.util import _mask_border_keypoints

from skimage.feature.censure_cy import _censure_dob_loop


OCTAGON_OUTER_SHAPE = [(5, 2), (5, 3), (7, 3), (9, 4), (9, 7), (13, 7),
                       (15, 10)]
OCTAGON_INNER_SHAPE = [(3, 0), (3, 1), (3, 2), (5, 2), (5, 3), (5, 4), (5, 5)]

STAR_SHAPE = [1, 2, 3, 4, 6, 8, 11, 12, 16, 22, 23, 32, 45, 46, 64, 90, 128]
STAR_FILTER_SHAPE = [(1, 0), (3, 1), (4, 2), (5, 3), (7, 4), (8, 5),
                     (9, 6),(11, 8), (13, 10), (14, 11), (15, 12), (16, 14)]


def _get_filtered_image(image, min_scale, max_scale, mode):

    scales = np.zeros((image.shape[0], image.shape[1],
                       max_scale - min_scale +1), dtype=np.double)

    if mode == 'dob':

        # make scales[:, :, i] contiguous memory block
        item_size = scales.itemsize
        scales.strides = (item_size * scales.shape[0],
                          item_size,
                          item_size * scales.shape[0] * scales.shape[1])

        integral_img = integral_image(image)

        for i in range(max_scale - min_scale + 1):
            n = min_scale + i

            # Constant multipliers for the outer region and the inner region
            # of the bilevel filters with the constraint of keeping the
            # DC bias 0.
            inner_weight = (1.0 / (2 * n + 1)**2)
            outer_weight = (1.0 / (12 * n**2 + 4 * n))

            _censure_dob_loop(n, integral_img, scales[:, :, i],
                              inner_weight, outer_weight)

    # NOTE : For the Octagon shaped filter, we implemented and evaluated the
    # slanted integral image based image filtering but the performance was
    # more or less equal to image filtering using
    # scipy.ndimage.filters.convolve(). Hence we have decided to use the
    # later for a much cleaner implementation.
    elif mode == 'octagon':
        # TODO : Decide the shapes of Octagon filters for scales > 7

        for i in range(max_scale - min_scale + 1):
            mo, no = OCTAGON_OUTER_SHAPE[min_scale + i - 1]
            mi, ni = OCTAGON_INNER_SHAPE[min_scale + i - 1]
            scales[:, :, i] = convolve(image,
                                       _octagon_filter_kernel(mo, no, mi, ni))
    elif mode == 'star':

        for i in range(max_scale - min_scale + 1):
            m = STAR_SHAPE[STAR_FILTER_SHAPE[min_scale + i - 1][0]]
            n = STAR_SHAPE[STAR_FILTER_SHAPE[min_scale + i - 1][1]]
            scales[:, :, i] = convolve(image,
                                       _star_filter_kernel(m, n))

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


def _octagon_filter_kernel(mo, no, mi, ni):
    outer = (mo + 2 * no)**2 - 2 * no * (no + 1)
    inner = (mi + 2 * ni)**2 - 2 * ni * (ni + 1)
    outer_weight = 1.0 / (outer - inner)
    inner_weight = 1.0 / inner
    c = ((mo + 2 * no) - (mi + 2 * ni)) // 2
    outer_oct = _oct(mo, no)
    inner_oct = np.zeros((mo + 2 * no, mo + 2 * no))
    inner_oct[c: -c, c: -c] = _oct(mi, ni)
    bfilter = (outer_weight * outer_oct -
               (outer_weight + inner_weight) * inner_oct)
    return bfilter


def _star(a):
    if a == 1:
        bfilter = np.zeros((3, 3))
        bfilter[:] = 1
        return bfilter
    m = 2 * a + 1
    n = a // 2
    selem_square = np.zeros((m + 2 * n, m + 2 * n), dtype=np.uint8)
    selem_square[n: m + n, n: m + n] = 1
    selem_triangle = np.zeros((m + 2 * n, m + 2 * n), dtype=np.uint8)
    selem_triangle[(m + 2 * n - 1) // 2, 0] = 1
    selem_triangle[(m + 1) // 2, n - 1] = 1
    selem_triangle[(m + 4 * n - 3) // 2, n - 1] = 1
    selem_triangle = convex_hull_image(selem_triangle).astype(int)
    selem_triangle += (selem_triangle[:, ::-1] + selem_triangle.T +
                       selem_triangle.T[::-1, :])
    return selem_square + selem_triangle


def _star_filter_kernel(m, n):
    c = m + m // 2 - n - n // 2
    outer_star = _star(m)
    inner_star = np.zeros_like(outer_star)
    inner_star[c: -c, c: -c] = _star(n)
    outer_weight = 1.0 / (np.sum(outer_star - inner_star))
    inner_weight = 1.0 / np.sum(inner_star)
    bfilter = (outer_weight * outer_star -
               (outer_weight + inner_weight) * inner_star)
    return bfilter


def _suppress_lines(feature_mask, image, sigma, line_threshold):
    Axx, Axy, Ayy = _compute_auto_correlation(image, sigma)
    feature_mask[(Axx + Ayy) * (Axx + Ayy)
                 > line_threshold * (Axx * Ayy - Axy * Axy)] = False


def keypoints_censure(image, min_scale=1, max_scale=7, mode='DoB',
                      non_max_threshold=0.15, line_threshold=10):
    """
    Extracts Censure keypoints along with the corresponding scale using
    either Difference of Boxes, Octagon or STAR bilevel filter.

    Parameters
    ----------
    image : 2D ndarray
        Input image.
    min_scale : positive integer
        Minimum scale to extract keypoints from.
    max_scale : positive integer
        Maximum scale to extract keypoints from. The keypoints will be
        extracted from all the scales except the first and the last i.e.
        from the scales in the range [min_scale + 1, max_scale - 1].
    mode : ('DoB', 'Octagon', 'STAR')
        Type of bilevel filter used to get the scales of the input image.
        Possible values are 'DoB', 'Octagon' and 'STAR'. The three modes
        represent the shape of the bilevel filters i.e. box(square), octagon
        and star respectively. For instance, a bilevel octagon filter consists
        of a smaller inner octagon and a larger outer octagon with the filter
        weights being uniformly negative in both the inner octagon while
        uniformly positive in the difference region. Use STAR and Octagon for
        better features and DoB for better performance.
    non_max_threshold : float
        Threshold value used to suppress maximas and minimas with a weak
        magnitude response obtained after Non-Maximal Suppression.
    line_threshold : float
        Threshold for rejecting interest points which have ratio of principal
        curvatures greater than this value.

    Returns
    -------
    keypoints : (N, 2) array
        Location of the extracted keypoints in the (row, col) format.
    scales : (N, 1) array
        The corresponding scale of the N extracted keypoints.

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
    # (1) First we generate the required scales on the input grayscale image
    # using a bilevel filter and stack them up in `filter_response`.
    # (2) We then perform Non-Maximal suppression in 3 x 3 x 3 window on the
    # filter_response to suppress points that are neither minima or maxima in
    # 3 x 3 x 3 neighbourhood. We obtain a boolean ndarray `feature_mask`
    # containing all the minimas and maximas in `filter_response` as True.
    # (3) Then we suppress all the points in the `feature_mask` for which the
    # corresponding point in the image at a particular scale has the ratio of
    # principal curvatures greater than `line_threshold`.
    # (4) Finally, we remove the border keypoints and return the keypoints
    # along with its corresponding scale.
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")

    mode = mode.lower()
    if mode not in ('dob', 'octagon', 'star'):
        raise ValueError('Mode must be one of "DoB", "Octagon", "STAR".')

    if max_scale - min_scale < 2:
        raise ValueError('The number of scales should be greater than or'
                         'equal to 3.')

    image = img_as_float(image)
    image = np.ascontiguousarray(image)

    # Generating all the scales
    filter_response = _get_filtered_image(image, min_scale, max_scale, mode)

    # Suppressing points that are neither minima or maxima in their 3 x 3 x 3
    # neighbourhood to zero
    minimas = minimum_filter(filter_response, (3, 3, 3)) == filter_response
    maximas = maximum_filter(filter_response, (3, 3, 3)) == filter_response

    feature_mask = minimas | maximas
    feature_mask[filter_response < non_max_threshold] = False

    for i in range(1, max_scale - min_scale):
        # sigma = (window_size - 1) / 6.0, so the window covers > 99% of the
        #                                  kernel's distribution
        # window_size = 7 + 2 * (min_scale - 1 + i)
        # Hence sigma = 1 + (min_scale - 1 + i)/ 3.0
        _suppress_lines(feature_mask[:, :, i], image,
                        (1 + (min_scale +  i - 1) / 3.0), line_threshold)

    rows, cols, scales = np.nonzero(feature_mask[..., 1:max_scale - min_scale])
    keypoints = np.column_stack([rows, cols])
    scales = scales + min_scale + 1

    if mode == 'dob':
        return keypoints, scales

    cumulative_mask = np.zeros(keypoints.shape[0], dtype=np.bool)

    if mode == 'octagon':
        for i in range(min_scale + 1, max_scale):
            c = (OCTAGON_OUTER_SHAPE[i - 1][0] - 1) // 2 \
                + OCTAGON_OUTER_SHAPE[i - 1][1]
            cumulative_mask |= _mask_border_keypoints(image, keypoints, c) \
                               & (scales == i)
    elif mode == 'star':
        for i in range(min_scale + 1, max_scale):
            c = STAR_SHAPE[STAR_FILTER_SHAPE[i - 1][0]] \
                + STAR_SHAPE[STAR_FILTER_SHAPE[i - 1][0]] // 2
            cumulative_mask |= _mask_border_keypoints(image, keypoints, c) \
                               & (scales == i)

    return keypoints[cumulative_mask], scales[cumulative_mask]
