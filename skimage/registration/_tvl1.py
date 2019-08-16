# coding: utf-8
"""TV-L1 optical flow algorithm implementation.

"""

from functools import partial
import numpy as np
from scipy import ndimage as ndi
from skimage.transform import warp

from .utils import coarse_to_fine


def _tvl1(I0, I1, flow0, dt, lambda_, tau, nwarp, niter, tol, prefilter):
    """TV-L1 solver for optical flow estimation.

    Parameters
    ----------
    I0 : ~numpy.ndarray
        The first gray scale image of the sequence.
    I1 : ~numpy.ndarray
        The second gray scale image of the sequence.
    flow0 : ~numpy.ndarray
        Vector field initialization.
    dt : float
        Time step of the numerical scheme. Convergence is proved for
        values dt < 0.125, but it can be larger for faster
        convergence.
    lambda_ : float
        Attachement parameter. The smaller this parameter is,
        the smoother is the solutions.
    tau : float
        Tightness parameter. It should have a small value in order to
        maintain attachement and regularization parts in
        correspondence.
    nwarp : int
        Number of times I1 is warped.
    niter : int
        Number of fixed point iteration.
    tol : float
        Tolerance used as stopping criterion based on the L² distance
        between two consecutive values of (u, v).
    prefilter : bool
        whether to prefilter the estimated optical flow before each
        image warp.

    Returns
    -------
    flow : ~numpy.ndarray
        The estimated optical flow.

    """

    grid = np.meshgrid(*[np.arange(n) for n in I0.shape], indexing='ij')

    f0 = lambda_ * tau
    f1 = dt / tau
    tol *= I0.size

    flow = flow0

    g = np.zeros((I0.ndim, ) + I0.shape)
    proj = np.zeros((I0.ndim, I0.ndim, ) + I0.shape)

    s_g = [slice(None), ] * g.ndim
    s_p = [slice(None), ] * proj.ndim
    s_d = [slice(None), ] * (proj.ndim-2)

    for _ in range(nwarp):
        if prefilter:
            flow = ndi.filters.median_filter(flow, [1]+I0.ndim*[3])

        wI1 = warp(I1, grid+flow, mode='nearest')
        grad = np.array(np.gradient(wI1))
        NI = (grad*grad).sum(0)
        NI[NI == 0] = 1

        rho_0 = wI1 - I0 - (grad*flow0).sum(0)

        for _ in range(niter):

            # Data term

            rho = rho_0 + (grad*flow).sum(0)

            idx = abs(rho) <= f0 * NI

            flow_ = flow

            flow_[:, idx] -= rho[idx]*grad[:, idx]/NI[idx]

            idx = ~idx
            srho = f0 * np.sign(rho[idx])
            flow_[:, idx] -= srho*grad[:, idx]

            # Regularization term
            flow = flow_.copy()

            for idx in range(flow.shape[0]):
                s_p[0] = idx
                for _ in range(2):
                    for ax in range(flow.shape[0]):
                        s_g[0] = ax
                        s_g[ax+1] = slice(0, -1)
                        g[tuple(s_g)] = np.diff(flow[idx], axis=ax)
                        s_g[ax+1] = slice(None)

                    norm = np.sqrt((g ** 2).sum(0))[np.newaxis, ...]
                    norm *= f1
                    norm += 1.
                    proj[idx] -= dt * g
                    proj[idx] /= norm

                    # d will be the (negative) divergence of p[idx]
                    d = -proj[idx].sum(0)
                    for ax in range(flow.shape[0]):
                        s_p[1] = ax
                        s_p[ax+2] = slice(0, -1)
                        s_d[ax] = slice(1, None)
                        d[tuple(s_d)] += proj[tuple(s_p)]
                        s_p[ax+2] = slice(None)
                        s_d[ax] = slice(None)

                    flow[idx] = flow_[idx] + d

        flow0 -= flow
        if (flow0*flow0).sum() < tol:
            break

        flow0 = flow

    return flow


def tvl1(I0, I1, dt=0.2, lambda_=15, tau=0.3, nwarp=5, niter=10,
         tol=1e-4, prefilter=False):
    """Coarse to fine TV-L1 optical flow estimator.

    TV-L1 ia popular algorithm for optical flow estimation intrudced
    by Zack et al. [1]_, improved in [2]_ and detailed in [3]_.

    Parameters
    ----------
    I0 : ~numpy.ndarray
        The first gray scale image of the sequence.
    I1 : ~numpy.ndarray
        The second gray scale image of the sequence.
    dt : float
        Time step of the numerical scheme. Convergence is proved for
        values dt < 0.125, but it can be larger for faster
        convergence.
    lambda_ : float
        Attachement parameter. The smaller this parameter is,
        the smoother is the solutions.
    tau : float
        Tightness parameter. It should have a small value in order to
        maintain attachement and regularization parts in
        correspondence.
    nwarp : int
        Number of times I1 is warped.
    niter : int
        Number of fixed point iteration.
    tol : float
        Tolerance used as stopping criterion based on the L² distance
        between two consecutive values of (u, v).
    prefilter : bool
        whether to prefilter the estimated optical flow before each
        image warp.

    Returns
    -------
    flow : tuple[~numpy.ndarray]
        The estimated optical flow.

    References
    ----------
    .. [1] Zach, C., Pock, T., & Bischof, H. (2007, September). A
       duality based approach for realtime TV-L 1 optical flow. In Joint
       pattern recognition symposium (pp. 214-223). Springer, Berlin,
       Heidelberg.
    .. [2] Wedel, A., Pock, T., Zach, C., Bischof, H., & Cremers,
       D. (2009). An improved algorithm for TV-L 1 optical flow. In
       Statistical and geometrical approaches to visual motion analysis
       (pp. 23-45). Springer, Berlin, Heidelberg.
    .. [3] Pérez, J. S., Meinhardt-Llopis, E., & Facciolo,
       G. (2013). TV-L1 optical flow estimation. Image Processing On
       Line, 2013, 137-150.

    Examples
    --------
    >>> from skimage.color import rgb2gray
    >>> from skimage.data import stereo_motorcycle
    >>> from skimage.registration import tvl1
    >>> I0, I1, disp = stereo_motorcycle()
    >>> # --- Convert the images to gray level: color is not supported.
    >>> I0 = rgb2gray(I0)
    >>> I1 = rgb2gray(I1)
    >>> flow = tvl1(I1, I0)

    """

    solver = partial(_tvl1, dt=dt, lambda_=lambda_, tau=tau,
                     nwarp=nwarp, niter=niter, tol=tol,
                     prefilter=prefilter)

    return coarse_to_fine(I0, I1, solver)
