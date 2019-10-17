import numpy as np
from ..transform import warp
from .._shared.utils import safe_as_int
from scipy.signal import get_window


def _rotational_mapping(output_coords, window_size, upsample_factor):
    """Docstring here
    """
    center = (window_size / 2) - 0.5
    rr = np.sqrt(((output_coords[:,0] - center) ** 2) + ((output_coords[:,1] - center) ** 2))
    rr = rr * upsample_factor
    cc = np.zeros_like(rr)
    coords = np.column_stack((cc, rr))
    return coords

def get_window2(window, size, upsample_factor=1):
    """Docstring here
    """
    Nx = size * upsample_factor
    w = get_window(window, Nx, fftbins=False)
    w = w[safe_as_int(np.floor(w.shape[0]/2)):, np.newaxis]
    win_im = np.column_stack((w, np.zeros_like(w)))
    map_args = {'window_size': size, 'upsample_factor': upsample_factor}
    window2d = warp(win_im, _rotational_mapping, map_args=map_args, output_shape=(size,size))
    return window2d
