import numpy as np
from ..transform import warp
from .._shared.utils import safe_as_int
from scipy.signal import get_window


def window(window_type, size, ndim=2, warp_kwargs=None):
    """Return an n-dimensional window of a given size and dimensionality.

    Parameters
    ----------
    window_type : string, float, or tuple
        The type of window to be created. Any window type supported by
        `scipy.signal.get_window` is allowed here. See notes below for a
        current list, or the SciPy documentation for the version of SciPy
        on your machine.
    size : int
        The size of the window along each axis (all axes will be equal length).
    ndim : int, optional (default: 2)
        The number of dimensions of the window.
    warp_kwargs : dict
        Keyword arguments passed to `transform.warp` (e.g.,
        `warp_kwargs={'order':3}` to change interpolation method).

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
    parameters that have to be supplied with the window name as a tuple
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

    Some coordinates in the output window will be outside of the original
    signal. These will be filled in according to the default behavior of
    `transform.warp` (i.e. zeros). This can be changed with the `mode`
    keyword argument passed to `transform.warp`.

    Window types:
    - boxcar
    - triang
    - blackman
    - hamming
    - hann
    - bartlett
    - flattop
    - parzen
    - bohman
    - blackmanharris
    - nuttall
    - barthann
    - kaiser (needs beta)
    - gaussian (needs standard deviation)
    - general_gaussian (needs power, width)
    - slepian (needs width)
    - dpss (needs normalized half-bandwidth)
    - chebwin (needs attenuation)
    - exponential (needs decay scale)
    - tukey (needs taper fraction)

    Examples
    --------
    Return a Hann window with shape (512, 512):

    >>> from skimage.filters import window
    >>> w = window('hann', 512)

    Return a Kaiser window with beta parameter of 16 and shape (256, 256, 256):

    >>> w = window(16, 256, ndim=3)

    Return a Tukey window with an alpha parameter of 0.8 and shape (100, 100):

    >>> w = window(('tukey', 0.8), 100)

    References
    ----------
    .. [1] Two-dimensional window design, Wikipedia,
           https://en.wikipedia.org/wiki/Two_dimensional_window_design
    """

    ndim = safe_as_int(ndim)
    size = safe_as_int(size)

    if ndim <= 0:
        raise ValueError("Number of dimensions must be greater than zero")

    w = get_window(window_type, size, fftbins=False)
    w = np.reshape(w, (-1,) + (1,) * (ndim-1))

    # Create coords for warping following `ndimage.map_coordinates` convention.
    L = [np.arange(size, dtype=np.double) for i in range(ndim)]
    coords = np.stack((np.meshgrid(*L)))
    center = (size / 2) - 0.5
    coords[0, ...] = np.sqrt(((coords - center) ** 2).sum(axis=0)) + center
    coords[1:, ...] = 0

    if warp_kwargs is None:
        warp_kwargs = {}

    return warp(w, coords, **warp_kwargs)
