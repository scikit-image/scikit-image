import numpy as np
from ..transform import warp
from scipy.signal import get_window as get_window1d


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
    # Only looking at center of window to right edge
    w = get_window1d(window, size, fftbins=False)
    w = w[int(np.floor(w.shape[0]/2)):]
    w = np.reshape(w, (-1,) + (1,) * (ndim-1))

    # Create coords for warping following `ndimage.map_coordinates` convention.
    L = [np.arange(size) for i in range(ndim)]
    coords = np.stack((np.meshgrid(*L)))
    center = (size / 2) - 0.5
    coords[0, ...] = np.sqrt(((coords - center) ** 2).sum(axis=0))
    coords[1:, ...] = 0
    return warp(w, coords)
