import numpy as np
from scipy import ndimage
from scipy import stats
from skimage.color import rgb2grey
from skimage.util import img_as_float
from skimage.feature import peak_local_max


def _compute_derivatives(image):
    """Compute derivatives in x and y direction using the Sobel operator.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    imx : ndarray
        Derivative in x-direction.
    imy : ndarray
        Derivative in y-direction.

    """

    imy = ndimage.sobel(image, axis=0, mode='constant', cval=0)
    imx = ndimage.sobel(image, axis=1, mode='constant', cval=0)

    return imx, imy


def _compute_auto_correlation(image, sigma):
    """Compute auto-correlation matrix using sum of squared differences.

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.

    Returns
    -------
    Axx : ndarray
        Element of the auto-correlation matrix for each pixel in input image.
    Axy : ndarray
        Element of the auto-correlation matrix for each pixel in input image.
    Ayy : ndarray
        Element of the auto-correlation matrix for each pixel in input image.

    """

    if image.ndim == 3:
        image = img_as_float(rgb2grey(image))

    imx, imy = _compute_derivatives(image)

    # structure tensore
    Axx = ndimage.gaussian_filter(imx * imx, sigma, mode='constant', cval=0)
    Axy = ndimage.gaussian_filter(imx * imy, sigma, mode='constant', cval=0)
    Ayy = ndimage.gaussian_filter(imy * imy, sigma, mode='constant', cval=0)

    return Axx, Axy, Ayy


def corner_kitchen_rosenfeld(image):
    """Compute Kitchen and Rosenfeld corner measure response image.

    The corner measure is calculated as follows::

        (imxx * imy**2 + imyy * imx**2 - 2 * imxy * imx * imy)
        ------------------------------------------------------
                        (imx**2 + imy**2)

    Where imx and imy are the first and imxx, imxy, imyy the second derivatives.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    response : ndarray
        Kitchen and Rosenfeld response image.

    """

    imx, imy = _compute_derivatives(image)
    imxx, imxy = _compute_derivatives(imx)
    imyx, imyy = _compute_derivatives(imy)

    numerator = (imxx * imy**2 + imyy * imx**2 - 2 * imxy * imx * imy)
    denominator = (imx**2 + imy**2)

    response = np.zeros_like(image, dtype=np.double)

    mask = denominator != 0
    response[mask] = numerator[mask] / denominator[mask]

    return response


def corner_harris(image, method='k', k=0.05, eps=1e-6, sigma=1):
    """Compute Harris corner measure response image.

    This corner detector uses information from the auto-correlation matrix A::

        A = [(imx**2)   (imx*imy)] = [Axx Axy]
            [(imx*imy)   (imy**2)]   [Axy Ayy]

    Where imx and imy are the first derivatives averaged with a gaussian filter.
    The corner measure is then defined as::

        det(A) - k * trace(A)**2

    or::

        2 * det(A) / (trace(A) + eps)

    Parameters
    ----------
    image : ndarray
        Input image.
    method : {'k', 'eps'}, optional
        Method to compute the response image from the auto-correlation matrix.
    k : float, optional
        Sensitivity factor to separate corners from edges, typically in range
        `[0, 0.2]`. Small values of k result in detection of sharp corners.
    eps : float, optional
        Normalisation factor (Noble's corner measure).
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.

    Returns
    -------
    response : ndarray
        Harris response image.

    References
    ----------
    .. [1] http://kiwi.cs.dal.ca/~dparks/CornerDetection/harris.htm
    .. [2] http://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from skimage.feature import corner_harris, corner_peaks
    >>> square = np.zeros([10, 10])
    >>> square[2:8, 2:8] = 1
    >>> square
    array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
    >>> corner_peaks(corner_harris(square), min_distance=1)
    array([[2, 2],
           [2, 7],
           [7, 2],
           [7, 7]])

    """

    Axx, Axy, Ayy = _compute_auto_correlation(image, sigma)

    # determinant
    detA = Axx * Ayy - Axy**2
    # trace
    traceA = Axx + Ayy

    if method == 'k':
        response = detA - k * traceA**2
    else:
        response = 2 * detA / (traceA + eps)

    return response


def corner_shi_tomasi(image, sigma=1):
    """Compute Shi-Tomasi (Kanade-Tomasi) corner measure response image.

    This corner detector uses information from the auto-correlation matrix A::

        A = [(imx**2)   (imx*imy)] = [Axx Axy]
            [(imx*imy)   (imy**2)]   [Axy Ayy]

    Where imx and imy are the first derivatives averaged with a gaussian filter.
    The corner measure is then defined as the smaller eigenvalue of A::

        ((Axx + Ayy) - sqrt((Axx - Ayy)**2 + 4 * Axy**2)) / 2

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.

    Returns
    -------
    response : ndarray
        Shi-Tomasi response image.

    References
    ----------
    .. [1] http://kiwi.cs.dal.ca/~dparks/CornerDetection/harris.htm
    .. [2] http://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from skimage.feature import corner_shi_tomasi, corner_peaks
    >>> square = np.zeros([10, 10])
    >>> square[2:8, 2:8] = 1
    >>> square
    array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
    >>> corner_peaks(corner_shi_tomasi(square), min_distance=1)
    array([[2, 2],
           [2, 7],
           [7, 2],
           [7, 7]])

    """

    Axx, Axy, Ayy = _compute_auto_correlation(image, sigma)

    # minimum eigenvalue of A
    response = ((Axx + Ayy) - np.sqrt((Axx - Ayy)**2 + 4 * Axy**2)) / 2

    return response


def corner_foerstner(image, sigma=1):
    """Compute Foerstner corner measure response image.

    This corner detector uses information from the auto-correlation matrix A::

        A = [(imx**2)   (imx*imy)] = [Axx Axy]
            [(imx*imy)   (imy**2)]   [Axy Ayy]

    Where imx and imy are the first derivatives averaged with a gaussian filter.
    The corner measure is then defined as::

        w = det(A) / trace(A)           (size of error ellipse)
        q = 4 * det(A) / trace(A)**2    (roundness of error ellipse)

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.

    Returns
    -------
    w : ndarray
        Error ellipse sizes.
    q : ndarray
        Roundness of error ellipse.

    References
    ----------
    .. [1] http://www.ipb.uni-bonn.de/uploads/tx_ikgpublication/foerstner87.fast.pdf
    .. [2] http://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from skimage.feature import corner_foerstner, corner_peaks
    >>> square = np.zeros([10, 10])
    >>> square[2:8, 2:8] = 1
    >>> square
    array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  1,  1,  1,  1,  1,  1,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
    >>> w, q = corner_foerstner(square)
    >>> accuracy_thresh = 0.5
    >>> roundness_thresh = 0.3
    >>> foerstner = (q > roundness_thresh) * (w > accuracy_thresh) * w
    >>> corner_peaks(foerstner, min_distance=1)
    array([[2, 2],
           [2, 7],
           [7, 2],
           [7, 7]])

    """

    Axx, Axy, Ayy = _compute_auto_correlation(image, sigma)

    # determinant
    detA = Axx * Ayy - Axy**2
    # trace
    traceA = Axx + Ayy

    w = np.zeros_like(image, dtype=np.double)
    q = np.zeros_like(image, dtype=np.double)

    mask = traceA != 0

    w[mask] = detA[mask] / traceA[mask]
    q[mask] = 4 * detA[mask] / traceA[mask]**2

    return w, q


def corner_subpix(image, corners, window_size=11, alpha=0.99):
    """Determine subpixel position of corners.

    Parameters
    ----------
    image : ndarray
        Input image.
    corners : (N, 2) ndarray
        Corner coordinates `(row, col)`.
    window_size : int, optional
        Search window size for subpixel estimation.
    alpha : float, optional
        Significance level for point classification.

    Returns
    -------
    positions : (N, 2) ndarray
        Subpixel corner positions. NaN for "not classified" corners.

    References
    ----------
    .. [1] http://www.ipb.uni-bonn.de/uploads/tx_ikgpublication/\
           foerstner87.fast.pdf
    .. [2] http://en.wikipedia.org/wiki/Corner_detection

    """

    # window extent in one direction
    wext = (window_size - 1) / 2

    # normal equation arrays
    N_dot = np.zeros((2, 2), dtype=np.double)
    N_edge = np.zeros((2, 2), dtype=np.double)
    b_dot = np.zeros((2, ), dtype=np.double)
    b_edge = np.zeros((2, ), dtype=np.double)

    # critical statistical test values
    redundancy = window_size**2 - 2
    t_crit_dot = stats.f.isf(1 - alpha, redundancy, redundancy)
    t_crit_edge = stats.f.isf(alpha, redundancy, redundancy)

    # coordinates of pixels within window
    y, x = np.mgrid[- wext:wext + 1, - wext:wext + 1]

    corners_subpix = np.zeros_like(corners, dtype=np.double)

    for i, (y0, x0) in enumerate(corners):

        # crop window around corner + border for sobel operator
        miny = y0 - wext - 1
        maxy = y0 + wext + 2
        minx = x0 - wext - 1
        maxx = x0 + wext + 2
        window = image[miny:maxy, minx:maxx]

        winx, winy = _compute_derivatives(window)

        # compute gradient suares and remove border
        winx_winx = (winx * winx)[1:-1, 1:-1]
        winx_winy = (winx * winy)[1:-1, 1:-1]
        winy_winy = (winy * winy)[1:-1, 1:-1]

        # sum of squared differences (mean instead of gaussian filter)
        Axx = np.sum(winx_winx)
        Axy = np.sum(winx_winy)
        Ayy = np.sum(winy_winy)

        # sum of squared differences weighted with coordinates
        # (mean instead of gaussian filter)
        bxx_x = np.sum(winx_winx * x)
        bxx_y = np.sum(winx_winx * y)
        bxy_x = np.sum(winx_winy * x)
        bxy_y = np.sum(winx_winy * y)
        byy_x = np.sum(winy_winy * x)
        byy_y = np.sum(winy_winy * y)

        # normal equations for subpixel position
        N_dot[0, 0] = Axx
        N_dot[0, 1] = N_dot[1, 0] = - Axy
        N_dot[1, 1] = Ayy

        N_edge[0, 0] = Ayy
        N_edge[0, 1] = N_edge[1, 0] = Axy
        N_edge[1, 1] = Axx

        b_dot[:] = bxx_y - bxy_x, byy_x - bxy_y
        b_edge[:] = byy_y + bxy_x, bxx_x + bxy_y

        # estimated positions
        est_dot = np.linalg.solve(N_dot, b_dot)
        est_edge = np.linalg.solve(N_edge, b_edge)

        # residuals
        ry_dot = y - est_dot[0]
        rx_dot = x - est_dot[1]
        ry_edge = y - est_edge[0]
        rx_edge = x - est_edge[1]
        # squared residuals
        rxx_dot = rx_dot * rx_dot
        rxy_dot = rx_dot * ry_dot
        ryy_dot = ry_dot * ry_dot
        rxx_edge = rx_edge * rx_edge
        rxy_edge = rx_edge * ry_edge
        ryy_edge = ry_edge * ry_edge

        # determine corner class (dot or edge)
        # variance for different models
        var_dot = np.sum(winx_winx * ryy_dot - 2 * winx_winy * rxy_dot \
                         + winy_winy * rxx_dot)
        var_edge = np.sum(winy_winy * ryy_edge + 2 * winx_winy * rxy_edge \
                          + winx_winx * rxx_edge)
        # test value (F-distributed)
        t = var_edge / var_dot
        # 1 for edge, -1 for dot, 0 for "not classified"
        corner_class = (t < t_crit_edge) - (t > t_crit_dot)

        if corner_class == - 1:
            corners_subpix[i, :] = y0 + est_dot[0], x0 + est_dot[1]
        elif corner_class == 0:
            corners_subpix[i, :] = np.nan, np.nan
        elif corner_class == 1:
            corners_subpix[i, :] = y0 + est_edge[0], x0 + est_edge[1]

    return corners_subpix


def corner_peaks(image, min_distance=10, threshold_abs=0, threshold_rel=0.1,
                 exclude_border=True, indices=True, num_peaks=np.inf,
                 footprint=None, labels=None):
    """Find corners in corner measure response image.

    This differs from `skimage.feature.peak_local_max` in that it suppresses
    multiple connected peaks with the same accumulator value.

    Parameters
    ----------
    See `skimage.feature.peak_local_max`.

    Returns
    -------
    See `skimage.feature.peak_local_max`.

    Examples
    --------
    >>> from skimage.feature import peak_local_max, corner_peaks
    >>> response = np.zeros((5, 5))
    >>> response[2:4, 2:4] = 1
    >>> response
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> peak_local_max(response, exclude_border=False)
    array([[2, 2],
           [2, 3],
           [3, 2],
           [3, 3]])
    >>> corner_peaks(response, exclude_border=False)
    array([[2, 2]])
    >>> corner_peaks(response, exclude_border=False, min_distance=0)
    array([[2, 2],
           [2, 3],
           [3, 2],
           [3, 3]])

    """

    peaks = peak_local_max(image, min_distance=min_distance,
                           threshold_abs=threshold_abs,
                           threshold_rel=threshold_rel,
                           exclude_border=exclude_border,
                           indices=False, num_peaks=num_peaks,
                           footprint=footprint, labels=labels)
    if min_distance > 0:
        coords = np.transpose(peaks.nonzero())
        for r, c in coords:
            if peaks[r, c]:
                peaks[r - min_distance:r + min_distance + 1,
                      c - min_distance:c + min_distance + 1] = False
                peaks[r, c] = True

    if indices is True:
        return np.transpose(peaks.nonzero())
    else:
        return peaks
