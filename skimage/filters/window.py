import numpy as np
from ..transform import warp
from .._shared.utils import safe_as_int
from scipy.signal import get_window


def _ndrotational_mapping(output_coords):
    """Mapping function for creating hyperspherically symmetric image.

    Parameters
    ----------
    output_coords : ndarray

    Returns
    -------
    coords : ndarray
    """
    window_size = output_coords.shape[1]
    center = (window_size / 2) - 0.5
    coords = np.zeros_like(output_coords, dtype=np.double)
    coords[0, ...] = np.sqrt(((output_coords - center) ** 2).sum(axis=0))
    return coords


def get_windownd(window, size, ndim=2):
    """Docstring here

    Parameters
    ----------
    window : string, float, or tuple
        The type of window to be created. Windows are based on
        `scipy.signal.get_window`.
    size : int
        The size of the window along each axis (all axes will be equal length).
    ndim : int
        The number of dimensions of the window.
    Returns
    -------

    Notes
    -----
    This function is based on `scipy.signal.get_window` and thus can access
    all of the window types available to that function.


    Examples
    --------
    """
    w = get_window(window, size, fftbins=False)
    w = w[int(np.floor(w.shape[0]/2)):]
    L = [np.arange(size) for i in range(ndim)]
    outcoords = np.stack((np.meshgrid(*L)))
    coords = _ndrotational_mapping(outcoords)
    for i in range(ndim-1):
        w = np.expand_dims(w, axis=1)
    windownd = warp(w, coords)
    return windownd
