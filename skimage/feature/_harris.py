"""
Harris corner detector

Inspired from Solem's implementation
http://www.janeriksolem.net/2009/01/harris-corner-detector-in-python.html
"""
from scipy import ndimage

from . import peak


def _compute_harris_response(image, eps=1e-6, gaussian_deviation=1):
    """Compute the Harris corner detector response function
    for each pixel in the image

    Parameters
    ----------
    image : ndarray of floats
        Input image.

    eps : float, optional
        Normalisation factor.

    gaussian_deviation : integer, optional
        Standard deviation used for the Gaussian kernel.

    Returns
    --------
    image : (M, N) ndarray
        Harris image response
    """
    if len(image.shape) == 3:
        image = image.mean(axis=2)

    # derivatives
    image = ndimage.gaussian_filter(image, gaussian_deviation)
    imx = ndimage.sobel(image, axis=0, mode='constant')
    imy = ndimage.sobel(image, axis=1, mode='constant')

    Wxx = ndimage.gaussian_filter(imx * imx, 1.5, mode='constant')
    Wxy = ndimage.gaussian_filter(imx * imy, 1.5, mode='constant')
    Wyy = ndimage.gaussian_filter(imy * imy, 1.5, mode='constant')

    # determinant and trace
    Wdet = Wxx * Wyy - Wxy**2
    Wtr = Wxx + Wyy
    # Alternate formula for Harris response.
    # Alison Noble, "Descriptions of Image Surfaces", PhD thesis (1989)
    harris = Wdet / (Wtr + eps)

    return harris


def harris(image, min_distance=10, threshold=0.1, eps=1e-6,
           gaussian_deviation=1):
    """Return corners from a Harris response image

    Parameters
    ----------
    image : ndarray of floats
        Input image.

    min_distance : int, optional
        Minimum number of pixels separating interest points and image boundary.

    threshold : float, optional
        Relative threshold impacting the number of interest points.

    eps : float, optional
        Normalisation factor.

    gaussian_deviation : integer, optional
        Standard deviation used for the Gaussian kernel.

    Returns
    -------
    coordinates : (N, 2) array
        (row, column) coordinates of interest points.

    Examples
    -------
    >>> square = np.zeros([10,10])
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
    >>> harris(square, min_distance=1)

    Corners of the square

    array([[3, 3],
           [3, 6],
           [6, 3],
           [6, 6]])
    """

    harrisim = _compute_harris_response(image, eps=eps,
                                        gaussian_deviation=gaussian_deviation)
    coordinates = peak.peak_local_max(harrisim, min_distance=min_distance,
                                      threshold_rel=threshold)
    return coordinates
