"""Global thresholding.

These algorithms create one or more scalar thresholds for an entire image.
"""

import inspect
import math
from collections import OrderedDict

import numpy as np
from scipy import ndimage as ndi

from .._shared.utils import warn
from .._shared.version_requirements import require
from ..exposure import histogram
from ._multiotsu_cy import (
    _get_multiotsu_thresh_indices,
    _get_multiotsu_thresh_indices_lut,
)


def _validate_image_histogram(image, hist, nbins=None, normalize=False):
    """Ensure that either image or hist were given, return valid histogram.

    If hist is given, image is ignored.

    Parameters
    ----------
    image : array or None
        Grayscale image.
    hist : array, 2-tuple of array, or None
        Histogram, either a 1D counts array, or an array of counts together
        with an array of bin centers.
    nbins : int, optional
        The number of bins with which to compute the histogram, if `hist` is
        None.
    normalize : bool
        If hist is not given, it will be computed by this function. This
        parameter determines whether the computed histogram is normalized
        (i.e. entries sum up to 1) or not.

    Returns
    -------
    counts : 1D array of float
        Each element is the number of pixels falling in each intensity bin.
    bin_centers : 1D array
        Each element is the value corresponding to the center of each intensity
        bin.

    Raises
    ------
    ValueError : if image and hist are both None
    """
    if image is None and hist is None:
        raise Exception("Either image or hist must be provided.")

    if hist is not None:
        if isinstance(hist, (tuple, list)):
            counts, bin_centers = hist
        else:
            counts = hist
            bin_centers = np.arange(counts.size)

        if counts[0] == 0 or counts[-1] == 0:
            # Trim histogram from both ends by removing starting and
            # ending zeroes as in histogram(..., source_range="image")
            cond = counts > 0
            start = np.argmax(cond)
            end = cond.size - np.argmax(cond[::-1])
            counts, bin_centers = counts[start:end], bin_centers[start:end]
    else:
        counts, bin_centers = histogram(
            image.reshape(-1), nbins, source_range='image', normalize=normalize
        )
    return counts.astype('float32', copy=False), bin_centers


def threshold_mean(image):
    """Return threshold value based on the mean of grayscale values.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] C. A. Glasbey, "An analysis of histogram-based thresholding
        algorithms," CVGIP: Graphical Models and Image Processing,
        vol. 55, pp. 532-537, 1993.
        :DOI:`10.1006/cgip.1993.1040`

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_mean(image)
    >>> binary = image > thresh
    """
    return np.mean(image)


def threshold_li(image, *, tolerance=None, initial_guess=None, iter_callback=None):
    """Compute threshold value by Li's iterative Minimum Cross Entropy method.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    tolerance : float, optional
        Finish the computation when the change in the threshold in an iteration
        is less than this value. By default, this is half the smallest
        difference between intensity values in ``image``.
    initial_guess : float or Callable[[array[float]], float], optional
        Li's iterative method uses gradient descent to find the optimal
        threshold. If the image intensity histogram contains more than two
        modes (peaks), the gradient descent could get stuck in a local optimum.
        An initial guess for the iteration can help the algorithm find the
        globally-optimal threshold. A float value defines a specific start
        point, while a callable should take in an array of image intensities
        and return a float value. Example valid callables include
        ``numpy.mean`` (default), ``lambda arr: numpy.quantile(arr, 0.95)``,
        or even :func:`skimage.filters.threshold_otsu`.
    iter_callback : Callable[[float], Any], optional
        A function that will be called on the threshold at every iteration of
        the algorithm.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
           Pattern Recognition, 26(4): 617-625
           :DOI:`10.1016/0031-3203(93)90115-D`
    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8): 771-776
           :DOI:`10.1016/S0167-8655(98)00057-9`
    .. [3] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165
           :DOI:`10.1117/1.1631315`
    .. [4] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_li(image)
    >>> binary = image > thresh
    """
    # Remove nan:
    image = image[~np.isnan(image)]
    if image.size == 0:
        return np.nan

    # Make sure image has more than one value; otherwise, return that value
    # This works even for np.inf
    if np.all(image == image.flat[0]):
        return image.flat[0]

    # At this point, the image only contains np.inf, -np.inf, or valid numbers
    image = image[np.isfinite(image)]
    # if there are no finite values in the image, return 0. This is because
    # at this point we *know* that there are *both* inf and -inf values,
    # because inf == inf evaluates to True. We might as well separate them.
    if image.size == 0:
        return 0.0

    # Li's algorithm requires positive image (because of log(mean))
    image_min = np.min(image)
    image -= image_min
    if image.dtype.kind in 'iu':
        tolerance = tolerance or 0.5
    else:
        tolerance = tolerance or np.min(np.diff(np.unique(image))) / 2

    # Initial estimate for iteration. See "initial_guess" in the parameter list
    if initial_guess is None:
        t_next = np.mean(image)
    elif callable(initial_guess):
        t_next = initial_guess(image)
    elif np.isscalar(initial_guess):  # convert to new, positive image range
        t_next = initial_guess - float(image_min)
        image_max = np.max(image) + image_min
        if not 0 < t_next < np.max(image):
            msg = (
                f'The initial guess for threshold_li must be within the '
                f'range of the image. Got {initial_guess} for image min '
                f'{image_min} and max {image_max}.'
            )
            raise ValueError(msg)
        t_next = image.dtype.type(t_next)
    else:
        raise TypeError(
            'Incorrect type for `initial_guess`; should be '
            'a floating point value, or a function mapping an '
            'array to a floating point value.'
        )

    # initial value for t_curr must be different from t_next by at
    # least the tolerance. Since the image is positive, we ensure this
    # by setting to a large-enough negative number
    t_curr = -2 * tolerance

    # Callback on initial iterations
    if iter_callback is not None:
        iter_callback(t_next + image_min)

    # Stop the iterations when the difference between the
    # new and old threshold values is less than the tolerance
    # or if the background mode has only one value left,
    # since log(0) is not defined.

    if image.dtype.kind in 'iu':
        hist, bin_centers = histogram(image.reshape(-1), source_range='image')
        hist = hist.astype('float32', copy=False)
        while abs(t_next - t_curr) > tolerance:
            t_curr = t_next
            foreground = bin_centers > t_curr
            background = ~foreground

            mean_fore = np.average(bin_centers[foreground], weights=hist[foreground])
            mean_back = np.average(bin_centers[background], weights=hist[background])

            if mean_back == 0:
                break

            t_next = (mean_back - mean_fore) / (np.log(mean_back) - np.log(mean_fore))

            if iter_callback is not None:
                iter_callback(t_next + image_min)

    else:
        while abs(t_next - t_curr) > tolerance:
            t_curr = t_next
            foreground = image > t_curr
            mean_fore = np.mean(image[foreground])
            mean_back = np.mean(image[~foreground])

            if mean_back == 0.0:
                break

            t_next = (mean_back - mean_fore) / (np.log(mean_back) - np.log(mean_fore))

            if iter_callback is not None:
                iter_callback(t_next + image_min)

    threshold = t_next + image_min
    return threshold


def threshold_otsu(image=None, *, nbins=256, hist=None):
    """Return threshold value based on Otsu's method.

    Either image or hist must be provided. If hist is provided, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray, optional
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    hist : array, or 2-tuple of arrays, optional
        Histogram from which to determine the threshold, and optionally a
        corresponding array of bin center intensities. If no hist provided,
        this function will compute it from the image.


    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method

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
    if image is not None and image.ndim > 2 and image.shape[-1] in (3, 4):
        warn(
            f'threshold_otsu is expected to work correctly only for '
            f'grayscale images; image shape {image.shape} looks like '
            f'that of an RGB image.'
        )

    # Check if the image has more than one intensity value; if not, return that
    # value
    if image is not None:
        first_pixel = image.reshape(-1)[0]
        if np.all(image == first_pixel):
            return first_pixel

    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold


def threshold_yen(image=None, *, nbins=256, hist=None):
    """Return threshold value based on Yen's method.
    Either image or hist must be provided. In case hist is given, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    hist : array, or 2-tuple of arrays, optional
        Histogram from which to determine the threshold, and optionally a
        corresponding array of bin center intensities.
        An alternative use of this function is to pass it only hist.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion
           for Automatic Multilevel Thresholding" IEEE Trans. on Image
           Processing, 4(3): 370-378. :DOI:`10.1109/83.366472`
    .. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165, :DOI:`10.1117/1.1631315`
           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
    .. [3] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_yen(image)
    >>> binary = image <= thresh
    """
    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    # On blank images (e.g. filled with 0) with int dtype, `histogram()`
    # returns ``bin_centers`` containing only one value. Speed up with it.
    if bin_centers.size == 1:
        return bin_centers[0]

    # Calculate probability mass function
    pmf = counts.astype('float32', copy=False) / counts.sum()
    P1 = np.cumsum(pmf)  # Cumulative normalized histogram
    P1_sq = np.cumsum(pmf**2)
    # Get cumsum calculated from end of squared array:
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    # P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid
    # '-inf' in crit. ImageJ Yen implementation replaces those values by zero.
    crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) * (P1[:-1] * (1.0 - P1[:-1])) ** 2)
    return bin_centers[crit.argmax()]


def threshold_minimum(image=None, *, nbins=256, max_num_iter=10000, hist=None):
    """Return threshold value based on minimum method.

    The histogram of the input ``image`` is computed if not provided and
    smoothed until there are only two maxima. Then the minimum in between is
    the threshold value.

    Either image or hist must be provided. In case hist is given, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray, optional
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    max_num_iter : int, optional
        Maximum number of iterations to smooth the histogram.
    hist : array, or 2-tuple of arrays, optional
        Histogram to determine the threshold from and a corresponding array
        of bin center intensities. Alternatively, only the histogram can be
        passed.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    Raises
    ------
    RuntimeError
        If unable to find two local maxima in the histogram or if the
        smoothing takes more than 1e4 iterations.

    References
    ----------
    .. [1] C. A. Glasbey, "An analysis of histogram-based thresholding
           algorithms," CVGIP: Graphical Models and Image Processing,
           vol. 55, pp. 532-537, 1993.
    .. [2] Prewitt, JMS & Mendelsohn, ML (1966), "The analysis of cell
           images", Annals of the New York Academy of Sciences 128: 1035-1053
           :DOI:`10.1111/j.1749-6632.1965.tb11715.x`

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_minimum(image)
    >>> binary = image > thresh
    """

    def find_local_maxima_idx(hist):
        # We can't use scipy.signal.argrelmax
        # as it fails on plateaus
        maximum_idxs = list()
        direction = 1

        for i in range(hist.shape[0] - 1):
            if direction > 0:
                if hist[i + 1] < hist[i]:
                    direction = -1
                    maximum_idxs.append(i)
            else:
                if hist[i + 1] > hist[i]:
                    direction = 1

        return maximum_idxs

    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    smooth_hist = counts.astype('float32', copy=False)

    for counter in range(max_num_iter):
        smooth_hist = ndi.uniform_filter1d(smooth_hist, 3)
        maximum_idxs = find_local_maxima_idx(smooth_hist)
        if len(maximum_idxs) < 3:
            break

    if len(maximum_idxs) != 2:
        raise RuntimeError('Unable to find two maxima in histogram')
    elif counter == max_num_iter - 1:
        raise RuntimeError('Maximum iteration reached for histogram' 'smoothing')

    # Find the lowest point between the maxima
    threshold_idx = np.argmin(smooth_hist[maximum_idxs[0] : maximum_idxs[1] + 1])

    return bin_centers[maximum_idxs[0] + threshold_idx]


def threshold_triangle(image, *, nbins=256):
    """Return threshold value based on the triangle algorithm.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Zack, G. W., Rogers, W. E. and Latt, S. A., 1977,
       Automatic Measurement of Sister Chromatid Exchange Frequency,
       Journal of Histochemistry and Cytochemistry 25 (7), pp. 741-753
       :DOI:`10.1177/25.7.70454`
    .. [2] ImageJ AutoThresholder code,
       http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_triangle(image)
    >>> binary = image > thresh
    """
    # nbins is ignored for integer arrays
    # so, we recalculate the effective nbins.
    hist, bin_centers = histogram(image.reshape(-1), nbins, source_range='image')
    nbins = len(hist)

    # Find peak, lowest and highest gray levels.
    arg_peak_height = np.argmax(hist)
    peak_height = hist[arg_peak_height]
    arg_low_level, arg_high_level = np.flatnonzero(hist)[[0, -1]]

    if arg_low_level == arg_high_level:
        # Image has constant intensity.
        return image.ravel()[0]

    # Flip is True if left tail is shorter.
    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = hist[::-1]
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1

    # If flip == True, arg_high_level becomes incorrect
    # but we don't need it anymore.
    del arg_high_level

    # Set up the coordinate system.
    width = arg_peak_height - arg_low_level
    x1 = np.arange(width)
    y1 = hist[x1 + arg_low_level]

    # Normalize.
    norm = np.sqrt(peak_height**2 + width**2)
    peak_height /= norm
    width /= norm

    # Maximize the length.
    # The ImageJ implementation includes an additional constant when calculating
    # the length, but here we omit it as it does not affect the location of the
    # minimum.
    length = peak_height * x1 - width * y1
    arg_level = np.argmax(length) + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    return bin_centers[arg_level]


def threshold_isodata(image=None, *, nbins=256, return_all=False, hist=None):
    """Return threshold value(s) based on ISODATA method.

    Histogram-based threshold, known as Ridler-Calvard method or inter-means.
    Threshold values returned satisfy the following equality::

        threshold = (image[image <= threshold].mean() +
                     image[image > threshold].mean()) / 2.0

    That is, returned thresholds are intensities that separate the image into
    two groups of pixels, where the threshold intensity is midway between the
    mean intensities of these groups.

    For integer images, the above equality holds to within one; for floating-
    point images, the equality holds to within the histogram bin-width.

    Either image or hist must be provided. In case hist is given, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    return_all : bool, optional
        If False (default), return only the lowest threshold that satisfies
        the above equality. If True, return all valid thresholds.
    hist : array, or 2-tuple of arrays, optional
        Histogram to determine the threshold from and a corresponding array
        of bin center intensities. Alternatively, only the histogram can be
        passed.

    Returns
    -------
    threshold : float or int or array
        Threshold value(s).

    References
    ----------
    .. [1] Ridler, TW & Calvard, S (1978), "Picture thresholding using an
           iterative selection method"
           IEEE Transactions on Systems, Man and Cybernetics 8: 630-632,
           :DOI:`10.1109/TSMC.1978.4310039`
    .. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165,
           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
           :DOI:`10.1117/1.1631315`
    .. [3] ImageJ AutoThresholder code,
           http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import coins
    >>> image = coins()
    >>> thresh = threshold_isodata(image)
    >>> binary = image > thresh
    """
    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    # image only contains one unique value
    if len(bin_centers) == 1:
        if return_all:
            return bin_centers
        else:
            return bin_centers[0]

    counts = counts.astype('float32', copy=False)

    # csuml and csumh contain the count of pixels in that bin or lower, and
    # in all bins strictly higher than that bin, respectively
    csuml = np.cumsum(counts)
    csumh = csuml[-1] - csuml

    # intensity_sum contains the total pixel intensity from each bin
    intensity_sum = counts * bin_centers

    # l and h contain average value of all pixels in that bin or lower, and
    # in all bins strictly higher than that bin, respectively.
    # Note that since exp.histogram does not include empty bins at the low or
    # high end of the range, csuml and csumh are strictly > 0, except in the
    # last bin of csumh, which is zero by construction.
    # So no worries about division by zero in the following lines, except
    # for the last bin, but we can ignore that because no valid threshold
    # can be in the top bin.
    # To avoid the division by zero, we simply skip over the last element in
    # all future computation.
    csum_intensity = np.cumsum(intensity_sum)
    lower = csum_intensity[:-1] / csuml[:-1]
    higher = (csum_intensity[-1] - csum_intensity[:-1]) / csumh[:-1]

    # isodata finds threshold values that meet the criterion t = (l + m)/2
    # where l is the mean of all pixels <= t and h is the mean of all pixels
    # > t, as calculated above. So we are looking for places where
    # (l + m) / 2 equals the intensity value for which those l and m figures
    # were calculated -- which is, of course, the histogram bin centers.
    # We only require this equality to be within the precision of the bin
    # width, of course.
    all_mean = (lower + higher) / 2.0
    bin_width = bin_centers[1] - bin_centers[0]

    # Look only at thresholds that are below the actual all_mean value,
    # for consistency with the threshold being included in the lower pixel
    # group. Otherwise, can get thresholds that are not actually fixed-points
    # of the isodata algorithm. For float images, this matters less, since
    # there really can't be any guarantees anymore anyway.
    distances = all_mean - bin_centers[:-1]
    thresholds = bin_centers[:-1][(distances >= 0) & (distances < bin_width)]

    if return_all:
        return thresholds
    else:
        return thresholds[0]


def threshold_multiotsu(image=None, *, classes=3, nbins=256, hist=None):
    r"""Generate `classes`-1 threshold values to divide gray levels in `image`,
    following Otsu's method for multiple classes.

    The threshold values are chosen to maximize the total sum of pairwise
    variances between the thresholded graylevel classes. See Notes and [1]_
    for more details.

    Either image or hist must be provided. If hist is provided, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray, optional
        Grayscale input image.
    classes : int, optional
        Number of classes to be thresholded, i.e. the number of resulting
        regions.
    nbins : int, optional
        Number of bins used to calculate the histogram. This value is ignored
        for integer arrays.
    hist : array, or 2-tuple of arrays, optional
        Histogram from which to determine the threshold, and optionally a
        corresponding array of bin center intensities. If no hist provided,
        this function will compute it from the image (see notes).

    Returns
    -------
    thresh : array
        Array containing the threshold values for the desired classes.

    Raises
    ------
    ValueError
         If ``image`` contains less grayscale value then the desired
         number of classes.

    Notes
    -----
    This implementation relies on a Cython function whose complexity
    is :math:`O\left(\frac{Ch^{C-1}}{(C-1)!}\right)`, where :math:`h`
    is the number of histogram bins and :math:`C` is the number of
    classes desired.

    If no hist is given, this function will make use of
    `skimage.exposure.histogram`, which behaves differently than
    `np.histogram`. While both allowed, use the former for consistent
    behaviour.

    The input image must be grayscale.

    References
    ----------
    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
           multilevel thresholding", Journal of Information Science and
           Engineering 17 (5): 713-727, 2001. Available at:
           <https://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>
           :DOI:`10.6688/JISE.2001.17.5.1`
    .. [2] Tosa, Y., "Multi-Otsu Threshold", a java plugin for ImageJ.
           Available at:
           <http://imagej.net/plugins/download/Multi_OtsuThreshold.java>

    Examples
    --------
    >>> from skimage.color import label2rgb
    >>> from skimage import data
    >>> image = data.camera()
    >>> thresholds = threshold_multiotsu(image)
    >>> regions = np.digitize(image, bins=thresholds)
    >>> regions_colorized = label2rgb(regions)
    """
    if image is not None and image.ndim > 2 and image.shape[-1] in (3, 4):
        warn(
            f'threshold_multiotsu is expected to work correctly only for '
            f'grayscale images; image shape {image.shape} looks like '
            f'that of an RGB image.'
        )

    # calculating the histogram and the probability of each gray level.
    prob, bin_centers = _validate_image_histogram(image, hist, nbins, normalize=True)
    prob = prob.astype('float32', copy=False)

    nvalues = np.count_nonzero(prob)
    if nvalues < classes:
        msg = (
            f'After discretization into bins, the input image has '
            f'only {nvalues} different values. It cannot be thresholded '
            f'in {classes} classes. If there are more unique values '
            f'before discretization, try increasing the number of bins '
            f'(`nbins`).'
        )
        raise ValueError(msg)
    elif nvalues == classes:
        thresh_idx = np.flatnonzero(prob)[:-1]
    else:
        # Get threshold indices
        try:
            thresh_idx = _get_multiotsu_thresh_indices_lut(prob, classes - 1)
        except MemoryError:
            # Don't use LUT if the number of bins is too large (if the
            # image is uint16 for example): in this case, the
            # allocated memory is too large.
            thresh_idx = _get_multiotsu_thresh_indices(prob, classes - 1)

    thresh = bin_centers[thresh_idx]

    return thresh


def _cross_entropy(image, *, threshold, bins=None):
    """Compute cross-entropy between distributions above and below a threshold.

    Parameters
    ----------
    image : array
        The input array of values.
    threshold : float
        The value dividing the foreground and background in ``image``.
    bins : int or array of float, optional
        The number of bins or the bin edges. (Any valid value to the ``bins``
        argument of ``np.histogram`` will work here.) For an exact calculation,
        each unique value should have its own bin. The default value for bins
        ensures exact handling of uint8 images: ``bins=256`` results in
        aliasing problems due to bin width not being equal to 1.

    Returns
    -------
    nu : float
        The cross-entropy target value as defined in [1]_.

    Notes
    -----
    See Li and Lee, 1993 [1]_; this is the objective function ``threshold_li``
    minimizes. This function can be improved but this implementation most
    closely matches equation 8 in [1]_ and equations 1-3 in [2]_.

    References
    ----------
    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
           Pattern Recognition, 26(4): 617-625
           :DOI:`10.1016/0031-3203(93)90115-D`
    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8): 771-776
           :DOI:`10.1016/S0167-8655(98)00057-9`
    """
    if bins is None:
        # Computing a histogram using np.histogram on a uint8 image with bins=256
        # doesn't work and results in aliasing problems. We use a fully specified set
        # of bins to ensure that each uint8 value false into its own bin.
        bins = np.arange(-0.5, 255.51, 1)

    histogram, bin_edges = np.histogram(image, bins=bins, density=True)
    bin_centers = np.convolve(bin_edges, [0.5, 0.5], mode='valid')
    t = np.flatnonzero(bin_centers > threshold)[0]
    m0a = np.sum(histogram[:t])  # 0th moment, background
    m0b = np.sum(histogram[t:])
    m1a = np.sum(histogram[:t] * bin_centers[:t])  # 1st moment, background
    m1b = np.sum(histogram[t:] * bin_centers[t:])
    mua = m1a / m0a  # mean value, background
    mub = m1b / m0b
    nu = -m1a * np.log(mua) - m1b * np.log(mub)
    return nu


def _try_all(image, methods=None, figsize=None, num_cols=2, verbose=True):
    """Returns a figure comparing the outputs of different methods.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    methods : dict, optional
        Names and associated functions.
        Functions must take and return an image.
    figsize : tuple, optional
        Figure size (in inches).
    num_cols : int, optional
        Number of columns.
    verbose : bool, optional
        Print function name for each method.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.
    """
    from matplotlib import pyplot as plt

    # Compute the image histogram for better performances
    nbins = 256  # Default in threshold functions
    hist = histogram(image.reshape(-1), nbins, source_range='image')

    # Handle default value
    methods = methods or {}

    num_rows = math.ceil((len(methods) + 1.0) / num_cols)
    fig, ax = plt.subplots(
        num_rows, num_cols, figsize=figsize, sharex=True, sharey=True
    )
    ax = ax.reshape(-1)

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')

    i = 1
    for name, func in methods.items():
        # Use precomputed histogram for supporting functions
        sig = inspect.signature(func)
        _kwargs = dict(hist=hist) if 'hist' in sig.parameters else {}

        ax[i].set_title(name)
        try:
            ax[i].imshow(func(image, **_kwargs), cmap=plt.cm.gray)
        except Exception as e:
            ax[i].text(
                0.5,
                0.5,
                f"{type(e).__name__}",
                ha="center",
                va="center",
                transform=ax[i].transAxes,
            )
        i += 1
        if verbose:
            print(func.__orifunc__)

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    return fig, ax


@require("matplotlib", ">=3.3")
def threshold_try_global(image, *, figsize=(8, 5), verbose=True):
    """Returns a figure comparing the outputs of different thresholding methods.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    figsize : tuple, optional
        Figure size (in inches).
    verbose : bool, optional
        Print function name for each method.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.

    Notes
    -----
    The following algorithms are used:

    * isodata
    * li
    * mean
    * minimum
    * otsu
    * triangle
    * yen

    Examples
    --------
    .. testsetup::
        >>> import pytest; _ = pytest.importorskip('matplotlib')

    >>> from skimage.data import text
    >>> fig, ax = threshold_try_global(text(), figsize=(10, 6), verbose=False)
    """

    def thresh(func):
        """
        A wrapper function to return a thresholded image.
        """

        def wrapper(im):
            return im > func(im)

        try:
            wrapper.__orifunc__ = func.__orifunc__
        except AttributeError:
            wrapper.__orifunc__ = func.__module__ + '.' + func.__name__
        return wrapper

    # Global algorithms.
    methods = OrderedDict(
        {
            'Isodata': thresh(threshold_isodata),
            'Li': thresh(threshold_li),
            'Mean': thresh(threshold_mean),
            'Minimum': thresh(threshold_minimum),
            'Otsu': thresh(threshold_otsu),
            'Triangle': thresh(threshold_triangle),
            'Yen': thresh(threshold_yen),
        }
    )

    return _try_all(image, figsize=figsize, methods=methods, verbose=verbose)
