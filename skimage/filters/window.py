import numpy as np
from ..transform import warp
from scipy.signal import get_window as get_window1d


def _ndrotational_mapping(output_coords):
    """Mapping function for creating a hyperspherically symmetric image.

    This function generates the mapping coordinates that can be used to
    warp a 1D array into n-dimensional space with hyperspherical symmetry,
    assuming the starting array is itself symmetric. To do this, it calculates
    the Euclidean distance from center for each position in the output array.

    Parameters
    ----------
    output_coords : ndarray
        Coordinate array that follows the `ndimage.map_coordinates` convention.
        The length of the first axis is equal to the number of dimensions of
        the image to be warped, and the remaining axes have the same shape as
        the image.

    Returns
    -------
    coords : ndarray
        Array of the same shape as `output_coords`, containing the Euclidean
        distance from center for a given point. For example, to warp an array
        of length 13 into 2-dimensions, `coords.shape` will be `(2, 13, 13)`
        and `coords[:, 9, 10]` will return `array([5., 0.])`.
    """
    window_size = output_coords.shape[1]
    center = (window_size / 2) - 0.5
    coords = np.zeros_like(output_coords, dtype=np.double)
    coords[0, ...] = np.sqrt(((output_coords - center) ** 2).sum(axis=0))
    return coords


def get_window(window, size, ndim=2):
    """Return an n-dimensional window of a given size and dimensionality.

    Parameters
    ----------
    window : string, float, or tuple
        The type of window to be created. Windows are based on
        `scipy.signal.get_window`.
    size : int
        The size of the window along each axis (all axes will be equal length).
    ndim : int, optional (default: 2)
        The number of dimensions of the window.

    Returns
    -------
    nd_window : ndarray
        A window of of `ndim` dimensions with a length of `size`
        along each axis.

    Notes
    -----
    This function is based on `scipy.signal.get_window` and thus can access
    all of the window types available to that function
    (e.g., `"hann"`, `"boxcar"`). Note that certain window types require
    parameters that have to supplied with the window name as a tuple
    (e.g., `("tukey", 0.8)`). If only a float is supplied, it is interpreted
    as the beta parameter of the kaiser window.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html
    for more details.

    Note that this function can generate very large arrays that can consume
    a large amount of available memory. For example, if `size=512` and
    `ndim=4`, the function will attempt to return an > 8.5GB array.

    Examples
    --------
    Return a Hann window with shape (512, 512):

    >>> from skimage.filters import get_window
    >>> w = get_window('hann', 512)

    Return a Kaiser window with beta parameter of 16 and shape (256, 256, 256):

    >>> w = get_window(16, 256, ndim=3)

    Return a Tukey window with an alpha parameter of 0.8 and shape (100, 100):

    >>> w = get_window(('tukey', 0.8), 100)
    """
    w = get_window1d(window, size, fftbins=False)
    w = w[int(np.floor(w.shape[0]/2)):]
    L = [np.arange(size) for i in range(ndim)]
    outcoords = np.stack((np.meshgrid(*L)))
    coords = _ndrotational_mapping(outcoords)
    w = np.reshape(w, (-1,) + (1,) * (ndim-1))
    return warp(w, coords)
