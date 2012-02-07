from __future__ import division

__all__ = ['structural_similarity']

import numpy as np
from numpy.lib import stride_tricks

from ..util.dtype import dtype_range

def _as_windows(X, win_size=7, flatten_first_axis=True):
    """Re-stride an array to simulate a sliding window.

    Parameters
    ----------
    X : 2D-ndarray
        Input image.

    Returns
    -------
    window : (N, M, win_size, win_size) ndarray
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

    return windows


def structural_similarity(X, Y, win_size=7, gradient=False, dynamic_range=None):
    """Compute the mean structural similarity index between two images.

    Parameters
    ----------
    X, Y : (N,N) ndarray
        Images.
    win_size : int
        The side-length of the sliding window used in comparison.  Must
        be an odd value.
    gradient : bool
        If True, also return the gradient.
    dynamic_range : int
        Dynamic range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from
        the image data-type.

    Returns
    -------
    s : float
        Strucutural similarity.
    grad : (N * N,) ndarray
        Gradient of the structural similarity index between X and Y.
        This is only returned if `gradient` is set to True.

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
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if dynamic_range is None:
        dmin, dmax = dtype_range[X.dtype.type]
        dynamic_range = dmax - dmin

    XW = _as_windows(X, win_size=win_size)
    YW = _as_windows(Y, win_size=win_size)

    NS = len(XW)
    NP = win_size * win_size

    ux = np.mean(np.mean(XW, axis=2), axis=2)
    uy = np.mean(np.mean(YW, axis=2), axis=2)

    # Compute variances var(X), var(Y) and var(X, Y)
    cov_norm = 1 / (win_size**2 - 1)
    XWM = XW - ux[..., None, None]
    YWM = YW - uy[..., None, None]
    vx = cov_norm * np.sum(np.sum(XWM**2, axis=2), axis=2)
    vy = cov_norm * np.sum(np.sum(YWM**2, axis=2), axis=2)
    vxy = cov_norm * np.sum(np.sum(XWM * YWM, axis=2), axis=2)

    R = dynamic_range
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * R)**2
    C2 = (K2 * R)**2

    A1, A2, B1, B2 = (v[..., None, None] for v in
                      (2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux**2 + uy**2 + C1,
                       vx + vy + C2))

    S = np.mean((A1 * A2) / (B1 * B2))

    if gradient:
        local_grad = 2 / (NP * B1**2 * B2**2) * \
            (
            A1 * B1 * (B2 * XW - A2 * YW) - \
            B1 * B2 * (A2 - A1) * ux[..., None, None] + \
            A1 * A2 * (B1 - B2) * uy[..., None, None]
            )

        grad = np.zeros_like(X, dtype=float)
        OW = _as_windows(grad, win_size=win_size)

        OW += local_grad
        grad /= NS

        return S, grad

    else:
        return S
