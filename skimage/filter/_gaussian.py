from scipy import ndimage
from skimage.util import img_as_float

__all__ = ['gaussian_filter']


def gaussian_filter(image, sigma, output=None, mode='nearest', cval=0,
                    multichannel=False):
    """
    Multi-dimensional Gaussian filter

    Parameters
    ----------

    image : array-like
        input image to filter
    sigma : scalar or sequence of scalars
        standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    output : array, optional
        The ``output`` parameter passes an array in which to store the
        filter output.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'nearest'.
    cval : scalar, optional
        Value to fill past edges of input if ``mode`` is 'constant'. Default
        is 0.0
    multichannel : bool, optional (default: False)
        Whether the last axis of the image is to be interpreted as multiple
        channels. If True, each channel is filtered separately (channels are
        not mixed together).


    Returns
    -------

    filtered_image : ndarray
        the filtered array

    Notes
    -----

    This function is a wrapper around :func:`scipy.ndimage.gaussian_filter`.

    Integer arrays are converted to float.

    The multi-dimensional filter is implemented as a sequence of
    one-dimensional convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.

    Examples
    --------

    >>> a = np.zeros((3, 3))
    >>> a[1, 1] = 1
    >>> a
    array([[ 0.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  0.]])
    >>> gaussian_filter(a, sigma=0.4)  # mild smoothing
    array([[ 0.00163116,  0.03712502,  0.00163116],
           [ 0.03712502,  0.84496158,  0.03712502],
           [ 0.00163116,  0.03712502,  0.00163116]])
    >>> gaussian_filter(a, sigma=1)  # more smooting
    array([[ 0.05855018,  0.09653293,  0.05855018],
           [ 0.09653293,  0.15915589,  0.09653293],
           [ 0.05855018,  0.09653293,  0.05855018]])
    >>> # Several modes are possible for handling boundaries
    >>> gaussian_filter(a, sigma=1, mode='reflect')
    array([[ 0.08767308,  0.12075024,  0.08767308],
           [ 0.12075024,  0.16630671,  0.12075024],
           [ 0.08767308,  0.12075024,  0.08767308]])
    >>> # For RGB images, each is filtered separately
    >>> from skimage.data import lena
    >>> image = lena()
    >>> filtered_lena = gaussian_filter(image, sigma=1, multichannel=True)
    """
    if multichannel:
        # do not filter across channels
        ndim = image.ndim
        sigma = [sigma] * (ndim - 1) + [0]
    image = img_as_float(image)
    return ndimage.gaussian_filter(image, sigma, mode=mode, cval=cval)
