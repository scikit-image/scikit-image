__all__ = ['threshold_adaptive',
           'threshold_otsu',
           'threshold_yen',
           'threshold_isodata',
           'threshold_li',
           'threshold_multiotsu']

import numpy as np
from scipy import ndimage as ndi
from ..exposure import histogram
from .._shared.utils import assert_nD, warn


def threshold_adaptive(image, block_size, method='gaussian', offset=0,
                       mode='reflect', param=None):
    """Applies an adaptive threshold to an array.

    Also known as local or dynamic thresholding where the threshold value is
    the weighted mean for the local neighborhood of a pixel subtracted by a
    constant. Alternatively the threshold can be determined dynamically by a a
    given function using the 'generic' method.

    Parameters
    ----------
    image : (N, M) ndarray
        Input image.
    block_size : int
        Odd size of pixel neighborhood which is used to calculate the
        threshold value (e.g. 3, 5, 7, ..., 21, ...).
    method : {'generic', 'gaussian', 'mean', 'median'}, optional
        Method used to determine adaptive threshold for local neighbourhood in
        weighted mean image.

        * 'generic': use custom function (see `param` parameter)
        * 'gaussian': apply gaussian filter (see `param` parameter for custom\
                      sigma value)
        * 'mean': apply arithmetic mean filter
        * 'median': apply median rank filter

        By default the 'gaussian' method is used.
    offset : float, optional
        Constant subtracted from weighted mean of neighborhood to calculate
        the local threshold value. Default offset is 0.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
        Default is 'reflect'.
    param : {int, function}, optional
        Either specify sigma for 'gaussian' method or function object for
        'generic' method. This functions takes the flat array of local
        neighbourhood as a single argument and returns the calculated
        threshold for the centre pixel.

    Returns
    -------
    threshold : (N, M) ndarray
        Thresholded binary image

    References
    ----------
    .. [1] http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#adaptivethreshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()[:50, :50]
    >>> binary_image1 = threshold_adaptive(image, 15, 'mean')
    >>> func = lambda arr: arr.mean()
    >>> binary_image2 = threshold_adaptive(image, 15, 'generic', param=func)
    """
    if block_size % 2 == 0:
        raise ValueError("The kwarg ``block_size`` must be odd! Given "
                         "``block_size`` {0} is even.".format(block_size))
    assert_nD(image, 2)
    thresh_image = np.zeros(image.shape, 'double')
    if method == 'generic':
        ndi.generic_filter(image, param, block_size,
                           output=thresh_image, mode=mode)
    elif method == 'gaussian':
        if param is None:
            # automatically determine sigma which covers > 99% of distribution
            sigma = (block_size - 1) / 6.0
        else:
            sigma = param
        ndi.gaussian_filter(image, sigma, output=thresh_image, mode=mode)
    elif method == 'mean':
        mask = 1. / block_size * np.ones((block_size,))
        # separation of filters to speedup convolution
        ndi.convolve1d(image, mask, axis=0, output=thresh_image, mode=mode)
        ndi.convolve1d(thresh_image, mask, axis=1,
                       output=thresh_image, mode=mode)
    elif method == 'median':
        ndi.median_filter(image, block_size, output=thresh_image, mode=mode)

    return image > (thresh_image - offset)


def threshold_otsu(image, nbins=256):
    """Return threshold value based on Otsu's method.

    Parameters
    ----------
    image : array
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels intensities that less or equal of
        this value assumed as foreground.

    References
    ----------
    .. [1] Wikipedia, http://en.wikipedia.org/wiki/Otsu's_Method

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh

    Notes
    -----
    The input image must be grayscale.
    """
    if image.shape[-1] in (3, 4):
        msg = "threshold_otsu is expected to work correctly only for " \
              "grayscale images; image shape {0} looks like an RGB image"
        warn(msg.format(image.shape))

    # Check if the image is multi-colored or not
    if image.min() == image.max():
        raise TypeError("threshold_otsu is expected to work with images " \
                        "having more than one color. The input image seems " \
                        "to have just one color {0}.".format(image.min()))

    hist, bin_centers = histogram(image.ravel(), nbins)
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


def threshold_yen(image, nbins=256):
    """Return threshold value based on Yen's method.

    Parameters
    ----------
    image : array
        Input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels intensities that less or equal of
        this value assumed as foreground.

    References
    ----------
    .. [1] Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion
           for Automatic Multilevel Thresholding" IEEE Trans. on Image
           Processing, 4(3): 370-378
    .. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165,
           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
    .. [3] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_yen(image)
    >>> binary = image <= thresh
    """
    hist, bin_centers = histogram(image.ravel(), nbins)
    # On blank images (e.g. filled with 0) with int dtype, `histogram()`
    # returns `bin_centers` containing only one value. Speed up with it.
    if bin_centers.size == 1:
        return bin_centers[0]

    # Calculate probability mass function
    pmf = hist.astype(np.float32) / hist.sum()
    P1 = np.cumsum(pmf)  # Cumulative normalized histogram
    P1_sq = np.cumsum(pmf ** 2)
    # Get cumsum calculated from end of squared array:
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    # P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid '-inf'
    # in crit. ImageJ Yen implementation replaces those values by zero.
    crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) *
                  (P1[:-1] * (1.0 - P1[:-1])) ** 2)
    return bin_centers[crit.argmax()]


def threshold_isodata(image, nbins=256, return_all=False):
    """Return threshold value(s) based on ISODATA method.

    Histogram-based threshold, known as Ridler-Calvard method or inter-means.
    Threshold values returned satisfy the following equality:

    `threshold = (image[image <= threshold].mean() +`
                 `image[image > threshold].mean()) / 2.0`

    That is, returned thresholds are intensities that separate the image into
    two groups of pixels, where the threshold intensity is midway between the
    mean intensities of these groups.

    For integer images, the above equality holds to within one; for floating-
    point images, the equality holds to within the histogram bin-width.

    Parameters
    ----------
    image : array
        Input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    return_all: bool, optional
        If False (default), return only the lowest threshold that satisfies
        the above equality. If True, return all valid thresholds.

    Returns
    -------
    threshold : float or int or array
        Threshold value(s).

    References
    ----------
    .. [1] Ridler, TW & Calvard, S (1978), "Picture thresholding using an
           iterative selection method"
    .. [2] IEEE Transactions on Systems, Man and Cybernetics 8: 630-632,
           http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4310039
    .. [3] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165,
           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
    .. [4] ImageJ AutoThresholder code,
           http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import coins
    >>> image = coins()
    >>> thresh = threshold_isodata(image)
    >>> binary = image > thresh
    """
    hist, bin_centers = histogram(image.ravel(), nbins)

    # image only contains one unique value
    if len(bin_centers) == 1:
        if return_all:
            return bin_centers
        else:
            return bin_centers[0]

    hist = hist.astype(np.float32)

    # csuml and csumh contain the count of pixels in that bin or lower, and
    # in all bins strictly higher than that bin, respectively
    csuml = np.cumsum(hist)
    csumh = np.cumsum(hist[::-1])[::-1] - hist

    # intensity_sum contains the total pixel intensity from each bin
    intensity_sum = hist * bin_centers

    # l and h contain average value of all pixels in that bin or lower, and
    # in all bins strictly higher than that bin, respectively.
    # Note that since exp.histogram does not include empty bins at the low or
    # high end of the range, csuml and csumh are strictly > 0, except in the
    # last bin of csumh, which is zero by construction.
    # So no worries about division by zero in the following lines, except
    # for the last bin, but we can ignore that because no valid threshold
    # can be in the top bin. So we just patch up csumh[-1] to not cause 0/0
    # errors.
    csumh[-1] = 1
    l = np.cumsum(intensity_sum) / csuml
    h = (np.cumsum(intensity_sum[::-1])[::-1] - intensity_sum) / csumh

    # isodata finds threshold values that meet the criterion t = (l + m)/2
    # where l is the mean of all pixels <= t and h is the mean of all pixels
    # > t, as calculated above. So we are looking for places where
    # (l + m) / 2 equals the intensity value for which those l and m figures
    # were calculated -- which is, of course, the histogram bin centers.
    # We only require this equality to be within the precision of the bin
    # width, of course.
    all_mean = (l + h) / 2.0
    bin_width = bin_centers[1] - bin_centers[0]

    # Look only at thresholds that are below the actual all_mean value,
    # for consistency with the threshold being included in the lower pixel
    # group. Otherwise can get thresholds that are not actually fixed-points
    # of the isodata algorithm. For float images, this matters less, since
    # there really can't be any guarantees anymore anyway.
    distances = all_mean - bin_centers
    thresholds = bin_centers[(distances >= 0) & (distances < bin_width)]

    if return_all:
        return thresholds
    else:
        return thresholds[0]


def threshold_li(image):
    """Return threshold value based on adaptation of Li's Minimum Cross Entropy method.

    Parameters
    ----------
    image : array
        Input image.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels intensities more than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
           Pattern Recognition, 26(4): 617-625
    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8): 771-776
    .. [3] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165
           http://citeseer.ist.psu.edu/sezgin04survey.html
    .. [4] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_li(image)
    >>> binary = image > thresh
    """
    # Copy to ensure input image is not modified
    image = image.copy()
    # Requires positive image (because of log(mean))
    immin = np.min(image)
    image -= immin
    imrange = np.max(image)
    tolerance = 0.5 * imrange / 256

    # Calculate the mean gray-level
    mean = np.mean(image)

    # Initial estimate
    new_thresh = mean
    old_thresh = new_thresh + 2 * tolerance

    # Stop the iterations when the difference between the
    # new and old threshold values is less than the tolerance
    while abs(new_thresh - old_thresh) > tolerance:
        old_thresh = new_thresh
        threshold = old_thresh + tolerance   # range
        # Calculate the means of background and object pixels
        mean_back = image[image <= threshold].mean()
        mean_obj = image[image > threshold].mean()

        temp = (mean_back - mean_obj) / (np.log(mean_back) - np.log(mean_obj))

        if temp < 0:
            new_thresh = temp - tolerance
        else:
            new_thresh = temp + tolerance

    return threshold + immin


def threshold_multiotsu(image, nclass=3, nbins=256):
    """Generates multiple thresholds for an input image. Based on the
    Multi-Otsu implementation by Liao and Chung.

    Parameters
    ----------
    image : array
        Grayscale input image.
    nclass : int, optional
        Number of classes to be thresholded, i.e. the number of resulting
        regions. Accepts an integer from 2 to 5.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    idx_thresh : array
        Array containing the threshold values for the desired classes.

    References
    ----------
    .. [1] Liao, P-S. and Chung, P-C., "A fast algorithm for multilevel
    thresholding", Journal of Information Science and Engineering 17
    (5): 713-727, 2001.

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_multiotsu(image)
    >>> region1 = image <= thresh[0]
    >>> region2 = (image > thresh[0]) & (image <= thresh[1])
    >>> region3 = image > thresh[1]
    """

    # check if the image is RGB.
    if image.shape[-1] in (3, 4):
        raise TypeError("Your image seems to be RGB (shape: {0}. Please use a"
                        "grayscale image.".format(image.shape))

    # check if the image has more than one color.
    if image.min() == image.max():
        raise TypeError("The input image seems to have just one color: {0}."
                        "Please use a grayscale image.".format(image.min()))

    # image needs to be treated as float.
    image = img_as_float(image)

    # calculating the histogram and the probability of each gray level.
    hist, _ = histogram(image.ravel(), nbins)
    prob = hist/np.size(image)

    gray_lvl = 256
    max_sigma = 0
    idx_thresh = np.zeros(nclass-1)
    momP, momS, var_btwcls = [np.zeros([gray_lvl, gray_lvl]) for n in range(3)]

    # building the lookup tables: calculating the first row.
    for u in range(1, gray_lvl):
        momP[1, u] = momP[1, u-1] + prob[u]
        momS[1, u] = momS[1, u-1] + (u)*prob[u]

    # building the lookup tables: calculating the other rows recursively.
    for u in range(2, gray_lvl):
        for v in range(u, gray_lvl):
            momP[u, v] = momP[1, v] - momP[1, u-1]
            momS[u, v] = momS[1, v] - momS[1, u-1]

    # building the lookup tables: calculating the between class variance.
    for u in range(1, gray_lvl):
        for v in range(u+1, gray_lvl):
            if (momP[u, v] != 0):
                var_btwcls[u, v] = (momS[u, v]**2)/momP[u, v]
            else:
                var_btwcls[u, v] = 0

    # finding max threshold candidates, depending on nclass.
    # number of thresholds is equal to number of classes - 1.
    if nclass == 2:
        for idx in range(gray_lvl - nclass):
            part_sigma = var_btwcls[1, idx] + var_btwcls[idx+1, gray_lvl-1]
            if max_sigma < part_sigma:
                idx_thresh[0] = idx
                max_sigma = part_sigma

    elif nclass == 3:
        for idx1 in range(gray_lvl - nclass):
            for idx2 in range(idx1+1, gray_lvl - nclass+1):
                part_sigma = var_btwcls[1, idx1] + \
                            var_btwcls[idx1+1, idx2] + \
                            var_btwcls[idx2+1, gray_lvl-1]

                if max_sigma < part_sigma:
                    idx_thresh[0] = idx1
                    idx_thresh[1] = idx2
                    max_sigma = part_sigma

    elif nclass == 4:
        for idx1 in range(gray_lvl - nclass):
            for idx2 in range(idx1+1, gray_lvl - nclass+1):
                for idx3 in range(idx2+1, gray_lvl - nclass+2):
                    part_sigma = var_btwcls[1, idx1] + \
                                var_btwcls[idx1+1, idx2] + \
                                var_btwcls[idx2+1, idx3] + \
                                var_btwcls[idx3+1, gray_lvl-1]

                    if max_sigma < part_sigma:
                        idx_thresh[0] = idx1
                        idx_thresh[1] = idx2
                        idx_thresh[2] = idx3
                        max_sigma = part_sigma

    elif nclass == 5:
        for idx1 in range(gray_lvl - nclass):
            for idx2 in range(idx1+1, gray_lvl - nclass+1):
                for idx3 in range(idx2+1, gray_lvl - nclass+2):
                    for idx4 in range(idx3+1, gray_lvl - nclass+3):
                        part_sigma = var_btwcls[1, idx1] + \
                            var_btwcls[idx1+1, idx2] + \
                            var_btwcls[idx2+1, idx3] + \
                            var_btwcls[idx3+1, idx4] + \
                            var_btwcls[idx4+1, gray_lvl-1]

                        if max_sigma < part_sigma:
                            idx_thresh[0] = idx1
                            idx_thresh[1] = idx2
                            idx_thresh[2] = idx3
                            idx_thresh[3] = idx4
                            max_sigma = part_sigma

    return idx_thresh.astype(int)
