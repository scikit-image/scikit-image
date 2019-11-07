import numpy as np
from ..transform import warp
from .._shared.utils import safe_as_int
from scipy.signal import get_window as get_window1d


def get_window(window, size, ndim=2, **kwargs):
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
    **kwargs : keyword arguments
        Passed to `transform.warp` (e.g., `order=3` to change interpolation
        method).

    Returns
    -------
    nd_window : ndarray
        A window of of `ndim` dimensions with a length of `size`
        along each axis. `dtype` is `np.double`.

    Notes
    -----
    This function is based on `scipy.signal.get_window` and thus can access
    all of the window types available to that function
    (e.g., `"hann"`, `"boxcar"`). Note that certain window types require
    parameters that have to supplied with the window name as a tuple
    (e.g., `("tukey", 0.8)`). If only a float is supplied, it is interpreted
    as the beta parameter of the Kaiser window.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html
    for more details.

    Note that this function can generate very large arrays that can consume
    a large amount of available memory. For example, if `size=512` and
    `ndim=4` are given as parameters to `filters.get_window`, it will attempt
    to return an > 8.5GB array.

    The approach taken here to create nD windows is first calculate the
    Euclidean distance from the center of the intended nD window to each
    position in the array. Then it uses that distance to sample, with
    interpolation, from a 1D window returned from `scipy.signal.get_window`.
    The method of interpolation can be changed with the `order` keyword
    argument passed to `transform.warp`.

    Examples
    --------
    Return a Hann window with shape (512, 512):

    >>> from skimage.filters import get_window
    >>> w = get_window('hann', 512)

    Return a Kaiser window with beta parameter of 16 and shape (256, 256, 256):

    >>> w = get_window(16, 256, ndim=3)

    Return a Tukey window with an alpha parameter of 0.8 and shape (100, 100):

    >>> w = get_window(('tukey', 0.8), 100)

    References
    ----------
    .. [1] Two-dimensional window design, Wikipedia,
           https://en.wikipedia.org/wiki/Two_dimensional_window_design
    """

    ndim = safe_as_int(ndim)
    size = safe_as_int(size)

    if ndim <= 0:
        raise ValueError("Number of dimensions must be greater than zero")

    # Only looking at center of window to right edge
    w = get_window1d(window, size, fftbins=False)
    w = w[int(np.floor(w.shape[0]/2)):]
    w = np.reshape(w, (-1,) + (1,) * (ndim-1))

    # Create coords for warping following `ndimage.map_coordinates` convention.
    L = [np.arange(size, dtype=np.double) for i in range(ndim)]
    coords = np.stack((np.meshgrid(*L)))
    center = (size / 2) - 0.5
    coords[0, ...] = np.sqrt(((coords - center) ** 2).sum(axis=0))
    if size % 2 == 0:
        coords[0, ...] = coords[0, ...] - 0.5
    coords[1:, ...] = 0
    return warp(w, coords, **kwargs)
