import numpy as np
from scipy import ndimage
from skimage.color import rgb2grey
from skimage.util import img_as_float
from . import peak, _interest


def moravec(image, block_size=3, mode='constant', cval=0):
    """Compute Moravec response image.

    This interest operator is comparatively fast but not rotation invariant.

    Parameters
    ----------
    image : ndarray
        Input image.
    block_size : int, optional
        Block size for mean filtering the squared gradients.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : double, optional
        Constant value to use for constant mode.

    Returns
    -------
    coordinates : (N, 2) array
        `(row, column)` coordinates of interest points.

    Examples
    -------
    >>> from skimage.feature import moravec, peak_local_max
    >>> square = np.zeros([10, 10])
    >>> square[1:9,1:9] = 1
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
    >>> moravec(square)), square.shape)
    (2, 6)

    """

    if image.ndim == 3:
        image =  rgb2grey(image)

    image = np.ascontiguousarray(img_as_float(image))

    return _corner._moravec(image, block_size)


def harris(image, eps=1e-6, gaussian_deviation=1):
    """Compute Harris response image.

    Parameters
    ----------
    image : ndarray of floats
        Input image.
    eps : float, optional
        Normalisation factor.
    gaussian_deviation : integer, optional
        Standard deviation used for the Gaussian kernel.

    Returns
    -------
    coordinates : (N, 2) array
        `(row, column)` coordinates of interest points.

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
    image = ndimage.gaussian_filter(image, gaussian_deviation, mode='constant',
                                    cval=0)
    imx = ndimage.sobel(image, axis=0, mode='constant', cval=0)
    imy = ndimage.sobel(image, axis=1, mode='constant', cval=0)

    Wxx = ndimage.gaussian_filter(imx * imx, 1.5, mode='constant', cval=0)
    Wxy = ndimage.gaussian_filter(imx * imy, 1.5, mode='constant', cval=0)
    Wyy = ndimage.gaussian_filter(imy * imy, 1.5, mode='constant', cval=0)

    # determinant and trace
    Wdet = Wxx * Wyy - Wxy**2
    Wtr = Wxx + Wyy

    # Alternate formula for Harris response.
    # Alison Noble, "Descriptions of Image Surfaces", PhD thesis (1989)
    harris = Wdet / (Wtr + eps)

    return harris
