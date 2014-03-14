__all__ = ['threshold_adaptive',
           'threshold_otsu',
           'threshold_yen',
           'threshold_isodata',
           'threshold_li', 
           'threshold_niblack',
           'threshold_sauvola',
           ]

import numpy as np
import scipy.ndimage
from ..exposure import histogram
from .._shared.utils import assert_nD
from skimage.exposure import histogram
from skimage.transform import integral_image


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
        Uneven size of pixel neighborhood which is used to calculate the
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
    assert_nD(image, 2)
    thresh_image = np.zeros(image.shape, 'double')
    if method == 'generic':
        scipy.ndimage.generic_filter(image, param, block_size,
                                     output=thresh_image, mode=mode)
    elif method == 'gaussian':
        if param is None:
            # automatically determine sigma which covers > 99% of distribution
            sigma = (block_size - 1) / 6.0
        else:
            sigma = param
        scipy.ndimage.gaussian_filter(image, sigma, output=thresh_image,
                                      mode=mode)
    elif method == 'mean':
        mask = 1. / block_size * np.ones((block_size,))
        # separation of filters to speedup convolution
        scipy.ndimage.convolve1d(image, mask, axis=0, output=thresh_image,
                                 mode=mode)
        scipy.ndimage.convolve1d(thresh_image, mask, axis=1,
                                 output=thresh_image, mode=mode)
    elif method == 'median':
        scipy.ndimage.median_filter(image, block_size, output=thresh_image,
                                    mode=mode)

    return image > (thresh_image - offset)


def threshold_otsu(image, nbins=256):
    """Return threshold value based on Otsu's method.

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
    .. [1] Wikipedia, http://en.wikipedia.org/wiki/Otsu's_Method

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh
    """
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
    # Requires positive image (because of log(mean))
    offset = image.min()
    # Can not use fixed tolerance for float image
    imrange = image.max() - offset
    image -= offset

    tolerance = 0.5 * imrange / 256.0
    # Calculate the mean gray-level
    mean = image.mean()

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

    return threshold + offset


def _mean_std(image, w):
    """Return local mean and standard deviation of each pixel using a
    neighborhood defined by window w x w. The algorithm uses Integral
    images to speedup computation. This is used by threshold_niblack
    and threshold_sauvola.

    Parameters
    ----------
    image : (N, M) ndarray
        Input image.
    w : int
        Odd window size w x w (e.g. 3, 5, 7, ..., 21, ...).

    Returns
    -------
    m : 2-D array of same size of image containning local mean values.
    s : 2-D array of same size of image containning local standard
        deviation values.

    References
    ----------
    .. [1] F. Shafait, D. Keysers, and T. M. Breuel, "Efficient
           implementation of local adaptive thresholding techniques
           using integral images." in Document Recognition and
           Retrieval XV, (San Jose, USA), Jan. 2008.
    """
    if w == 1 or w % 2 == 0:
        raise ValueError(
            "Window size w = %s must be odd and greater than 1." % (w))
    I = integral_image(image)  # Integral Image.

    # Pad left and top of Integral image with zeros
    I = np.vstack((np.zeros((1, I.shape[1]), I.dtype), I))
    I = np.hstack((np.zeros((I.shape[0], 1), I.dtype), I))

    kern = np.zeros((w + 1, w + 1))
    kern[0, 0], kern[-1, -1] = 1, 1
    kern[[0, -1], [-1, 0]] = -1
    # w2 holds total of pixels in window for each pixel (usually w x w).
    w2 = scipy.ndimage.convolve(
        np.ones(image.shape, np.float), np.ones((w, w)), mode='constant')

    m = scipy.ndimage.convolve(I, kern, mode='nearest')[:-1, :-1] / w2
    g = image.astype(np.float)
    g2 = g ** 2.
    m2 = m ** 2.
    sum_g2 = scipy.ndimage.convolve(g2, np.ones((w, w)), mode='constant')
    sum_m2 = w2 * m2
    s2 = (sum_g2 - sum_m2) / w2
    s = np.sqrt(s2)
    return m, s


def threshold_niblack(image, w=15, k=0.2, offset=0):
    """Applies Niblack local threshold to an array.

    A threshold T is calculated for every pixel in the image using the
    following formula:

    T = m(x,y) - k * s(x,y)

    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by window w x w centered around
    the pixel. k is a configurable parameter that weights the effect
    of standar deviation.

    Parameters
    ----------
    image: (N, M) ndarray
        Input image.
    w : int, optional
        Odd size of pixel neighborhood window w x w (e.g. 3, 5, 7,
        ..., 21, ...). Default: 15.
    k : float, optional
        Value of parameter k in threshold formula. Default: 0.2.
    offset : float, optional
        Constant subtracted from obtained local thresholds.
        Default: 0.0.

    Returns
    -------
    threshold : (N, M) ndarray
        Thresholded binary image

    References
    ----------
    .. [1] Niblack, W (1986), An introduction to Digital Image
           Processing, Prentice-Hall.

    Examples
    --------
    ... from skimage.data import page
    ... image = page()
    ... binary_image = threshold_niblack(image, w=7, k=0.1)
    """

    m, s = _mean_std(image, w)
    t = m - k * s
    return image > (t - offset)


def threshold_sauvola(image, method='sauvola', w=15, k=0.2, r=128., offset=0,
                      p=2, q=10):
    """Applies Sauvola local threshold to an array. Sauvola is a
    modification of Niblack technique.

    In the original method a threshold T is calculated for every pixel
    in the image using the following formula:

    T = m(x,y) * (1 + k * ((s(x,y) / R) - 1))

    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by window w x w centered around
    the pixel. k is a configurable parameter that weights the effect
    of standar deviation. R is the maximum standard deviation of
    a greyscale image (R = 128).

    In Wolf's variation the threshold T is given by:

    T = (1 - k) * m(x,y) + k * M + k * (s(x,y) / R) * (m(x,y) - M)

    where R is the maximum standard deviation found in all local
    neighborhoods and M is the minimum pixel intensity in image.

    In Phansalkar's variation image pixels are normalized and the
    threshold T is given by:

    T = m * (1 + p * exp(-q * m(x,y)) + k * ((s(x,y) / R) - 1))

    where p and q are fixed values of 2 and 10 respectively. The
    other parameters have the same meaning as in the previous methods.

    Parameters
    ----------
    image: (N, M) ndarray
        Input image.
    method : {'original', 'wolf', 'phansalkar'}, optional.
        method used for computing local thresholds.

        * 'sauvola': Uses original implementation described in [1].
        * 'wolf': Uses Wolf's variation described in [2].
        * 'phansalkar': Uses Phansalkar's variation described in [3]-

        Default: 'sauvola'.
    w : int, optional
        Odd size of pixel neighborhood window w x w (e.g. 3, 5, 7,
        ..., 21, ...). Default: 15.
    k : float, optional
        Value of parameter k in threshold formula. Default: 0.2.
    r : float, optional
        Value of R in threshold formula. Not used by method 'wolf'.
        Default: 128.
    offset : float, optional
        Constant subtracted from obtained local thresholds.
        Default: 0.0.
    p : float, optional
        Value of p in 'phansalkar' threshold formula. Default: 2.
    q : float, optional
        Value of q in 'phansalkar' threshold formula. Default: 10.

    Returns
    -------
    threshold : (N, M) ndarray
        Thresholded binary image

    References
    ----------
    .. [1] J. Sauvola and M. Pietikainen, "Adaptive document image
           binarization," Pattern Recognition 33(2),
           pp. 225-236, 2000.
    .. [2] C. Wolf, J-M. Jolion, "Extraction and Recognition of
           Artificial Text in Multimedia Documents", Pattern
           Analysis and Applications, 6(4):309-326, (2003).
    .. [3] Phansalskar, N; More, S & Sabale, A et al. (2011), "Adaptive
           local thresholding for detection of nuclei in diversity
           stained cytology images.", International Conference on
           Communications and Signal Processing (ICCSP): 218-220.

    Examples
    --------
    ... from skimage.data import page
    ... image = page()
    ... binary_sauvola = threshold_sauvola(image, method='sauvola',
                                           w=15, k=0.2, r=128)
    ... binary_wolf = threshold_sauvola(image, method='wolf',
                                        w=7, k=0.2)
    ... binary_phansalkar = threshold_sauvola(image, method='phansalkar',
                                              w=7, k=0.2, r=128)
    """

    t = np.zeros_like(image)
    m, s = _mean_std(image, w)
    if method == 'sauvola':
        t = m * (1 + k * ((s / r) - 1))
    elif method == 'wolf':
        R = s.max()  # Max std_dev used by Wolf.
        M = image.min()
        t = (1 - k) * m + k * M + k * (s / R) * (m - M)
    elif method == 'phansalkar':
        image = image.astype(np.float) / 255.  # Normalized image.
        m, s = _mean_std(image, w)
        rn = r / 255.
        t = m * (1 + p * np.exp(-q * m) + k * ((s / rn) - 1))
    else:
        raise ValueError("Unknown method: %s" % (method))

    return image > (t - offset)
