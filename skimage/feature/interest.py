import numpy as np
from scipy import ndimage
from skimage.color import rgb2grey
from . import peak


def harris(image, method='k', k=0.05, eps=1e-6, sigma=1):
    """Compute Harris response image.

    Parameters
    ----------
    image : ndarray
        Input image.
    method : {'k', 'eps'}, optional
        Method to
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

    if image.ndim == 3:
        image = rgb2grey(image)

    # derivatives
    imx = ndimage.sobel(image, axis=0, mode='constant', cval=0)
    imy = ndimage.sobel(image, axis=1, mode='constant', cval=0)

    # sum of squared differences / structure tensore
    Axx = ndimage.gaussian_filter(imx * imx, sigma,
                                  mode='constant', cval=0)
    Axy = ndimage.gaussian_filter(imx * imy, sigma,
                                  mode='constant', cval=0)
    Ayy = ndimage.gaussian_filter(imy * imy, sigma,
                                  mode='constant', cval=0)

    # determinant
    detA = Axx * Ayy - Axy**2
    # trace
    traceA = Axx + Ayy

    if method == 'k':
        harris = detA - k * traceA**2
    else:
        harris = 2 * detA / (traceA + eps)

    return harris
