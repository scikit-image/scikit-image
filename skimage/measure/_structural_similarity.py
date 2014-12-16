from __future__ import division

__all__ = ['structural_similarity']

import numpy as np

from ..util.dtype import dtype_range
from ..util.shape import view_as_windows


def structural_similarity(X, Y, win_size=7,
                          gradient=False, dynamic_range=None):
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
        Structural similarity.
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

    XW = view_as_windows(X, (win_size, win_size))
    YW = view_as_windows(Y, (win_size, win_size))

    NS = len(XW)
    NP = win_size * win_size

    ux = np.mean(np.mean(XW, axis=2), axis=2)
    uy = np.mean(np.mean(YW, axis=2), axis=2)

    # Compute variances var(X), var(Y) and var(X, Y)
    cov_norm = 1 / (win_size ** 2 - 1)
    XWM = XW - ux[..., None, None]
    YWM = YW - uy[..., None, None]
    vx = cov_norm * np.sum(np.sum(XWM ** 2, axis=2), axis=2)
    vy = cov_norm * np.sum(np.sum(YWM ** 2, axis=2), axis=2)
    vxy = cov_norm * np.sum(np.sum(XWM * YWM, axis=2), axis=2)

    R = dynamic_range
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = (v[..., None, None] for v in
                      (2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))

    S = np.mean((A1 * A2) / (B1 * B2))

    if gradient:
        local_grad = 2 / (NP * B1 ** 2 * B2 ** 2) * \
            (A1 * B1 * (B2 * XW - A2 * YW) -
             B1 * B2 * (A2 - A1) * ux[..., None, None] +
             A1 * A2 * (B1 - B2) * uy[..., None, None])

        grad = np.zeros_like(X, dtype=float)
        OW = view_as_windows(grad, (win_size, win_size))

        OW += local_grad
        grad /= NS

        return S, grad

    else:
        return S
