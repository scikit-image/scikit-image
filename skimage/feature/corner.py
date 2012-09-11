import numpy as np
from scipy import ndimage
from skimage.color import rgb2grey
from . import peak


def _compute_derivatives(image):
    """Compute derivatives in x and y direction using the Sobel operator.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    imx, imy : arrays
        Derivatives in x and y direction.

    """

    imx = ndimage.sobel(image, axis=0, mode='constant', cval=0)
    imy = ndimage.sobel(image, axis=1, mode='constant', cval=0)

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
    Axx, Axy, Ayy : arrays
        Elements of the auto-correlation matrix for each pixel in input image.

    """

    if image.ndim == 3:
        image = rgb2grey(image)

    imx, imy = _compute_derivatives(image)

    # structure tensore
    Axx = ndimage.gaussian_filter(imx * imx, sigma, mode='constant', cval=0)
    Axy = ndimage.gaussian_filter(imx * imy, sigma, mode='constant', cval=0)
    Ayy = ndimage.gaussian_filter(imy * imy, sigma, mode='constant', cval=0)

    return Axx, Axy, Ayy


def corner_kitchen_rosenfeld(image):
    """Compute Kitchen and Rosenfeld response image.

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

    response = (imxx * imy**2 + imyy * imx**2 - 2 * imxy * imx * imy) \
               / (imx**2 + imy**2)

    return response


def corner_harris(image, method='k', k=0.05, eps=1e-6, sigma=1):
    """Compute Harris response image.

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
    ..[1] http://kiwi.cs.dal.ca/~dparks/CornerDetection/harris.htm
    ..[2] http://en.wikipedia.org/wiki/Corner_detection

    Examples
    -------
    >>> from skimage.feature import harris, peak_local_max
    >>> square = np.zeros([10, 10])
    >>> square[2:8,2:8] = 1
    >>> square
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    >>> peak_local_max(harris(square), min_distance=1)
    array([[3, 3],
           [3, 6],
           [6, 3],
           [6, 6]])

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
    """Compute Shi-Tomasi (Kanade-Tomasi) response image.

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
    ..[1] http://kiwi.cs.dal.ca/~dparks/CornerDetection/harris.htm
    ..[2] http://en.wikipedia.org/wiki/Corner_detection

    Examples
    -------
    >>> from skimage.feature import shi_tomasi, peak_local_max
    >>> square = np.zeros([10, 10])
    >>> square[2:8,2:8] = 1
    >>> square
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    >>> peak_local_max(shi_tomasi(square), min_distance=1)
    array([[3, 3],
           [3, 6],
           [6, 3],
           [6, 6]])

    """

    Axx, Axy, Ayy = _compute_auto_correlation(image, sigma)

    # minimum eigenvalue of A
    response = ((Axx + Ayy) - np.sqrt((Axx - Ayy)**2 + 4 * Axy**2)) / 2

    return response
