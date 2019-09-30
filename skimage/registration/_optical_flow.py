# coding: utf-8
"""TV-L1 optical flow algorithm implementation.

"""

from functools import partial
import numpy as np
from scipy import ndimage as ndi
from skimage.transform import warp

from ._optical_flow_utils import coarse_to_fine


def _tvl1(image0, image1, flow0, attachment, tightness, nwarp, niter,
          tol, prefilter):
    """TV-L1 solver for optical flow estimation.

    Parameters
    ----------
    image0 : ndarray, shape (M, N[, P[, ...]])
        The first gray scale image of the sequence.
    image1 : ndarray, shape (M, N[, P[, ...]])
        The second gray scale image of the sequence.
    flow0 : ndarray, shape (image0.ndim, M, N[, P[, ...]])
        Initialization for the vector field.
    attachment : float
        Attachment parameter. The smaller this parameter is,
        the smoother is the solutions.
    tightness : float
        Tightness parameter. It should have a small value in order to
        maintain attachement and regularization parts in
        correspondence.
    nwarp : int
        Number of times image1 is warped.
    niter : int
        Number of fixed point iteration.
    tol : float
        Tolerance used as stopping criterion based on the L² distance
        between two consecutive values of (u, v).
    prefilter : bool
        Whether to prefilter the estimated optical flow before each
        image warp.

    Returns
    -------
    flow : ndarray, shape ((image0.ndim, M, N[, P[, ...]])
        The estimated optical flow components for each axis.

    """

    dtype = image0.dtype
    grid = np.meshgrid(*[np.arange(n, dtype=dtype) for n in image0.shape],
                       indexing='ij')

    dt = 0.5/image0.ndim
    reg_niter = 2
    f0 = attachment * tightness
    f1 = dt / tightness
    tol *= image0.size

    flow_current = flow_previous = flow0

    g = np.zeros((image0.ndim, ) + image0.shape, dtype=dtype)
    proj = np.zeros((image0.ndim, image0.ndim, ) + image0.shape,
                    dtype=dtype)

    s_g = [slice(None), ] * g.ndim
    s_p = [slice(None), ] * proj.ndim
    s_d = [slice(None), ] * (proj.ndim-2)

    for _ in range(nwarp):
        if prefilter:
            flow_current = ndi.median_filter(flow_current,
                                             [1]+image0.ndim*[3])

        image1_warp = warp(image1, grid+flow_current, mode='nearest')
        grad = np.array(np.gradient(image1_warp))
        NI = (grad*grad).sum(0)
        NI[NI == 0] = 1

        rho_0 = image1_warp - image0 - (grad*flow_current).sum(0)

        for _ in range(niter):

            # Data term

            rho = rho_0 + (grad*flow_current).sum(0)

            idx = abs(rho) <= f0 * NI

            flow_auxiliary = flow_current

            flow_auxiliary[:, idx] -= rho[idx]*grad[:, idx]/NI[idx]

            idx = ~idx
            srho = f0 * np.sign(rho[idx])
            flow_auxiliary[:, idx] -= srho*grad[:, idx]

            # Regularization term
            flow_current = flow_auxiliary.copy()

            for idx in range(image0.ndim):
                s_p[0] = idx
                for _ in range(reg_niter):
                    for ax in range(image0.ndim):
                        s_g[0] = ax
                        s_g[ax+1] = slice(0, -1)
                        g[tuple(s_g)] = np.diff(flow_current[idx], axis=ax)
                        s_g[ax+1] = slice(None)

                    norm = np.sqrt((g ** 2).sum(0))[np.newaxis, ...]
                    norm *= f1
                    norm += 1.
                    proj[idx] -= dt * g
                    proj[idx] /= norm

                    # d will be the (negative) divergence of proj[idx]
                    d = -proj[idx].sum(0)
                    for ax in range(image0.ndim):
                        s_p[1] = ax
                        s_p[ax+2] = slice(0, -1)
                        s_d[ax] = slice(1, None)
                        d[tuple(s_d)] += proj[tuple(s_p)]
                        s_p[ax+2] = slice(None)
                        s_d[ax] = slice(None)

                    flow_current[idx] = flow_auxiliary[idx] + d

        flow_previous -= flow_current  # The difference as stopping criteria
        if (flow_previous*flow_previous).sum() < tol:
            break

        flow_previous = flow_current

    return flow_current


def optical_flow_tvl1(image0, image1, *, attachment=15, tightness=0.3,
                      nwarp=5, niter=10, tol=1e-4, prefilter=False,
                      single_precision=True):
    r"""Coarse to fine optical flow estimator.

    The TV-L1 solver is applied at each level of the image
    pyramid. TV-L1 is a popular algorithm for optical flow estimation
    introduced by Zack et al. [1]_, improved in [2]_ and detailed in [3]_.

    Parameters
    ----------
    image0 : ndarray, shape (M, N[, P[, ...]])
        The first gray scale image of the sequence.
    image1 : ndarray, shape (M, N[, P[, ...]])
        The second gray scale image of the sequence.
    attachment : float
        Attachment parameter (:math:`\lambda` in [1]_). The smaller
        this parameter is, the smoother the returned result will be.
    tightness : float
        Tightness parameter (:math:`\tau` in [1]_). It should have
        a small value in order to maintain attachement and
        regularization parts in correspondence.
    nwarp : int
        Number of times image1 is warped.
    niter : int
        Number of fixed point iteration.
    tol : float
        Tolerance used as stopping criterion based on the L² distance
        between two consecutive values of (u, v).
    prefilter : bool
        Whether to prefilter the estimated optical flow before each
        image warp. This helps to remove the potential outliers.
    single_precision : bool
        If True, single precision float is used. Double precision
        float is used otherwise. Single precision provides good
        results and saves memory usage and computation time compared
        to double precision.

    Returns
    -------
    flow : ndarray, shape ((image0.ndim, M, N[, P[, ...]])
        The estimated optical flow components for each axis.

    Notes
    -----
    Color images are not supported.

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
    >>> from skimage.registration import optical_flow_tvl1
    >>> image0, image1, disp = stereo_motorcycle()
    >>> # --- Convert the images to gray level: color is not supported.
    >>> image0 = rgb2gray(image0)
    >>> image1 = rgb2gray(image1)
    >>> flow = optical_flow_tvl1(image1, image0)

    """

    solver = partial(_tvl1, attachment=attachment,
                     tightness=tightness, nwarp=nwarp, niter=niter,
                     tol=tol, prefilter=prefilter)

    return coarse_to_fine(image0, image1, solver,
                          single_precision=single_precision)
