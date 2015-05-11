from __future__ import division

__all__ = ['structural_similarity']

import numpy as np
from scipy.ndimage.filters import uniform_filter, convolve1d

from ..util.dtype import dtype_range


def gaussian_filter2(X, sigma=1.5, size=11):
    """ nD Gaussian filter with specific window extent.

    matches the implementation of Wang. et. al.

    Parameters
    ----------
    X : ndarray
        image
    sigma : float
        Gaussian standard deviation (pixels)
    size : float
        Gaussian kernel extent (pixels)

    Returns
    -------
    X : ndarray
        filtered image

    Notes
    -----
    scipy.ndimage.gaussian is very similar, but uses a 13 tap FIR filter
    rather than the 11 tap one of Wang. et. al.
    """
    radius = (size - 1) // 2
    r = np.arange(2*radius + 1) - radius
    filt = np.exp(-(r * r)/(2 * sigma * sigma))
    filt /= filt.sum()
    for ax in range(X.ndim):
        X = convolve1d(X, filt, axis=ax)
    return X


def _discard_edges(X, pad):
    """ Remove border of width pad from ndarray X.

    Parameters
    ----------
    X : ndarray
        image
    pad : int or sequence of ints
        border width to remove.  Can be a list of values corresponding to each
        axis.  If pad is an integer, the same width is removed from all axes.

    Returns
    -------
    Y : nadarray
        image with edges removed

    """
    X = np.asanyarray(X)
    if pad == 0:
        return X

    if isinstance(pad, int):
        slice_array = [slice(pad, -pad), ] * X.ndim
    else:
        if len(pad) != X.ndim:
            raise ValueError("pad array must match number of X dimensions")
        slice_array = []
        for d in range(X.ndim):
            slice_array.append(slice(pad[d], -pad[d]))

    return X[slice_array]


def structural_similarity(X, Y, win_size=None, gradient=False,
                          dynamic_range=None, multichannel=None,
                          gaussian_weights=False, full=False,
                          image_content_weighting=False, **kwargs):
    """Compute the mean structural similarity index between two images.

    Parameters
    ----------
    X, Y : ndarray
        Images.
    win_size : int or None
        The side-length of the sliding window used in comparison.  Must be an
        odd value.  Default is 11 if `gaussian_weights` is True, 7 otherwise.
    gradient : bool
        If True, also return the gradient.
    dynamic_range : int
        Dynamic range of the input image (distance between minimum and maximum
        possible values).  By default, this is estimated from the image
        data-type.
    multichannel : int or None
        If True, treat the last dimension of the array as channels. Similarity
        calculations are done independently for each channel then averaged.
        Defaults to True only if X is 3D and X.shape[2] == 3.
    gaussian_weights : bool
        If True, each patch (of size win_size) has its mean and variance
        spatially weighted by a normalized Gaussian kernel of width sigma=1.5.
    full : bool
        If True, return the full structural similarity image instead of the
        mean value
    image_content_weighting : bool
        If True, weight the ssim mean is spatially weighted by image content as
        proposed in Wang and Shang 2006 [3].

    Other Parameters
    ----------------
    use_sample_covariance : bool
        if True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        algorithm parameter, K1 (small constant, see [1])
    K2 : float
        algorithm parameter, K2 (small constant, see [1])
    sigma : float
        sigma for the Gaussian when `gaussian_weights` is True.

    Returns
    -------
    mssim : float or ndarray
        mean structural similarity.
    grad : ndarray
        Gradient of the structural similarity index between X and Y [2]. This
        is only returned if `gradient` is set to True.
    S : ndarray
        Full SSIM image.  This is only returned if `full` is set to True.

    Notes
    -----
    To exactly match the implementation of Wang et. al. [1], set
    `gaussian_weights` to True, `win_size` to 11, and `use_sample_covariance`
    to False.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.

    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.

    .. [3] Wang, Z. and Shang, X.  Spatial pooling strategies for perceptual
       image quality assessment. Proc. IEEE Inter. Conf. Image. Proc.
       2945-2948.
    """
    if not X.dtype == Y.dtype:
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if image_content_weighting and gradient:
        raise ValueError(
            "gradient not implemented for image content weighted case")

    # default treats 3D arrays with shape[2] == 3 as multichannel
    if multichannel is None:
        if X.ndim == 3 and X.shape[2] == 3:
            multichannel = True
        else:
            multichannel = False

    if multichannel:
        # loop over channels
        args = locals()
        args.pop('X')
        args.pop('Y')
        args['multichannel'] = False
        nch = X.shape[-1]
        mssim = np.empty(nch)
        if gradient:
            G = np.empty(X.shape)
        if full:
            S = np.empty(X.shape)
        for ch in range(nch):
            ch_result = structural_similarity(X[..., ch], Y[..., ch], **args)
            if gradient and full:
                mssim[..., ch], G[..., ch], S[..., ch] = ch_result
            elif gradient:
                mssim[..., ch], G[..., ch] = ch_result
            elif full:
                mssim[..., ch], S[..., ch] = ch_result
            else:
                mssim[..., ch] = ch_result
        mssim = mssim.mean()
        if gradient and full:
            return mssim, G, S
        elif gradient:
            return mssim, G
        elif full:
            return mssim, S
        else:
            return mssim

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if win_size is None:
        if gaussian_weights:
            win_size = 11  # 11 to match Wang et. al. 2004
        else:
            win_size = 7   # backwards compatibility

    if np.any((np.asarray(X.shape) - win_size) < 0):
        raise ValueError("win_size exceeds image extent")

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if dynamic_range is None:
        dmin, dmax = dtype_range[X.dtype.type]
        dynamic_range = dmax - dmin

    ndim = X.ndim

    if gaussian_weights:
        # sigma = 1.5 to match Wang et. al. 2004
        filter_func = gaussian_filter2
        filter_args = {'sigma': sigma, 'size': win_size}
    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    uxy = filter_func(X * Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = dynamic_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim
    if image_content_weighting:
        # weight with Eq. 7 of Wang and Simoncelli 2006.
        W = np.log((1 + vx / C2) * (1 + vy / C2))
        W /= W.sum()
        mssim = _discard_edges(S * W, pad).sum()
    else:
        mssim = _discard_edges(S, pad).mean()

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = filter_func(A1 / D, **filter_args) * X
        grad += filter_func(-S / B2, **filter_args) * Y
        grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D,
                            **filter_args)
        grad *= (2 / X.size)

        if full:
            return mssim, grad, S
        else:
            return mssim, grad
    else:
        if full:
            return mssim, S
        else:
            return mssim
