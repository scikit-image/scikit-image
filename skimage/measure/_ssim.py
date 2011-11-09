from __future__ import division

__all__ = ['ssim']

import numpy as np
from numpy.lib import stride_tricks

def _as_windows(X, win_size=7):
    """Re-stride an array to simulate a sliding window.

    Parameters
    ----------
    X : 2D-ndarray
        Input image.

    Returns
    -------
    window : (N, win_size, win_size) ndarray
        Sliding windows.

    """
    if not X.ndim == 2:
        raise ValueError('Input images must be 2-dimensional.')

    X = np.ascontiguousarray(X)
    r, c = X.shape

    strides = X.strides
    row_jump, el_jump = strides
    half_width = (win_size // 2)

    new_strides = (row_jump, el_jump, row_jump, el_jump)
    new_rows = r - 2 * half_width
    new_cols = c - 2 * half_width
    new_shape = (new_rows, new_cols, win_size, win_size)

    windows = stride_tricks.as_strided(X, shape=new_shape, strides=new_strides)
    windows = windows.reshape((-1, win_size, win_size))

    return windows


def ssim(X, Y, win_size=7, dynamic_range=255):
    """Compute the structural similarity index between two images.

    Parameters
    ----------
    X, Y : (N,N) ndarray
        Images.
    win_size : int
        The side-length of the sliding window used in comparison.  Must
        be an odd value.
    dynamic_range : int
        Dynamic range of the input image (distance between minimum and
        maximum possible values).  This should eventually be
        auto-computed, but just specifying it manually for now.

    Returns
    -------
    s : float
        Strucutural similarity.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.

    """
    if not X.dtype == Y.dtype:
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Inout images must have the same dimensions.')

    import time

    tic = time.time()

    XW = _as_windows(X, win_size=win_size)
    YW = _as_windows(Y, win_size=win_size)

    tic = time.time()
    
    # Flatten windows
    XW = XW.reshape(XW.shape[0], -1)
    YW = YW.reshape(YW.shape[0], -1)

    ux = np.mean(XW, axis=1)
    uy = np.mean(YW, axis=1)

    tic = time.time()

    # Compute variances var(X), var(Y) and var(X, Y)
    cov_norm = 1 / (win_size**2 - 1)
    XWM = XW - ux[:, None]
    YWM = YW - uy[:, None]
    vx = cov_norm * np.sum(XWM**2, axis=1)
    vy = cov_norm * np.sum(YWM**2, axis=1)
    vxy = cov_norm * np.sum(XWM * YWM, axis=1)    

    R = dynamic_range
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * R)**2
    C2 = (K2 * R)**2

    return np.mean(((2 * ux * uy + C1) * (2 * vxy + C2)) / \
                   ((ux**2 + uy**2 + C1) * (vx + vy + C2)))
