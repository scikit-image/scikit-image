# coding: utf-8
"""TV-L1 optical flow algorithm implementation.

"""

from functools import partial
import numpy as np
from scipy import ndimage as ndi
from skimage.transform import warp

from .utils import coarse_to_fine, central_diff, forward_diff, div


def _tvl1(I0, I1, u0, v0, dt, lambda_, tau, nwarp, niter, tol, prefilter):
    """TV-L1 solver for optical flow estimation.

    Parameters
    ----------
    I0 : ~numpy.ndarray
        The first gray scale image of the sequence.
    I1 : ~numpy.ndarray
        The second gray scale image of the sequence.
    u0 : ~numpy.ndarray
        Initialization for the horizontal component of the vector
        field.
    v0 : ~numpy.ndarray
        Initialization for the vertical component of the vector
        field.
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
    u, v : tuple[~numpy.ndarray]
        The horizontal and vertical components of the estimated
        optical flow.

    """

    nl, nc = I0.shape
    y, x = np.meshgrid(np.arange(nl), np.arange(nc), indexing='ij')

    f0 = lambda_*tau
    f1 = dt/tau

    u = u0.copy()
    v = v0.copy()

    pu1 = np.zeros_like(u0)
    pu2 = np.zeros_like(u0)
    pv1 = np.zeros_like(u0)
    pv2 = np.zeros_like(u0)

    for _ in range(nwarp):
        if prefilter:
            u = ndi.filters.median_filter(u, 3)
            v = ndi.filters.median_filter(v, 3)

        wI1 = warp(I1, np.array([y+v, x+u]), mode='nearest')
        Ix, Iy = central_diff(wI1)
        NI = Ix*Ix + Iy*Iy
        NI[NI == 0] = 1

        rho_0 = wI1 - I0 - u0*Ix - v0*Iy

        for __ in range(niter):

            # Data term

            rho = rho_0 + u*Ix + v*Iy

            idx = abs(rho) <= f0*NI

            u_ = u.copy()
            v_ = v.copy()

            u_[idx] -= rho[idx]*Ix[idx]/NI[idx]
            v_[idx] -= rho[idx]*Iy[idx]/NI[idx]

            idx = ~idx
            srho = f0*np.sign(rho[idx])
            u_[idx] -= srho*Ix[idx]
            v_[idx] -= srho*Iy[idx]

            # Regularization term

            for ___ in range(2):

                u = u_ - tau*div(pu1, pu2)

                ux, uy = forward_diff(u)
                ux *= f1
                uy *= f1
                Q = 1 + np.sqrt(ux*ux + uy*uy)

                pu1 += ux
                pu1 /= Q
                pu2 += uy
                pu2 /= Q

                v = v_ - tau*div(pv1, pv2)

                vx, vy = forward_diff(v)
                vx *= f1
                vy *= f1
                Q = 1 + np.sqrt(vx*vx + vy*vy)

                pv1 += vx
                pv1 /= Q
                pv2 += vy
                pv2 /= Q

        u0 -= u
        v0 -= v
        if (u0*u0+v0*v0).sum()/(u.size) < tol:
            break
        else:
            u0, v0 = u.copy(), v.copy()

    return u, v


def tvl1(I0, I1, dt=0.2, lambda_=15, tau=0.3, nwarp=5, niter=10,
         tol=1e-4, prefilter=False):
    """Coarse to fine TV-L1 optical flow estimator. A popular algorithm
    intrudced by Zack et al. [1]_, improved in [2]_ and detailed in [3]_.

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
    u, v : tuple[~numpy.ndarray]
        The horizontal and vertical components of the estimated
        optical flow.

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
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from skimage.color import rgb2gray
    >>> from skimage.data import stereo_motorcycle
    >>> from skimage.registration import tvl1
    >>> # --- Load the sequence
    >>> I0, I1, disp = stereo_motorcycle()
    >>> # --- Convert the images to gray level: color is not supported.
    >>> I0 = rgb2gray(I0)
    >>> I1 = rgb2gray(I1)
    >>> # --- Compute the optical flow
    >>> u, v = tvl1(I1, I0)
    >>> # --- Display the result
    >>> fig = plt.figure(figsize=(12, 4))
    >>> ax0, ax1 = fig.subplots(1, 2)
    >>> nl, nc = u.shape
    >>> step = 10
    >>> y, x = np.mgrid[:nl:step, :nc:step]
    >>> # Ground truth
    >>> gt_u = disp[::step, ::step]
    >>> gt_v = np.zeros_like(gt_u)
    >>> gt_norm = np.sqrt(disp*disp)
    >>> ax0.imshow(gt_norm)
    >>> ax0.set_aspect('equal')
    >>> ax0.quiver(x, y, gt_u, gt_v, units='dots',
                   angles='xy', scale_units='xy')
    >>> ax0.set_title('Ground truth')
    >>> ax0.set_axis_off()
    >>> # Estimated optical flow
    >>> u_ = u[::step, ::step]
    >>> v_ = v[::step, ::step]
    >>> of_norm = np.sqrt(u*u+v*v)
    >>> ax1.imshow(of_norm)
    >>> ax1.set_aspect('equal')
    >>> ax1.quiver(x, y, u_, v_, units='dots',
                   angles='xy', scale_units='xy')
    >>> ax1.set_title('Estimated optical flow')
    >>> ax1.set_axis_off()
    >>> fig.set_tight_layout(True)
    >>> plt.show()

    """

    solver = partial(_tvl1, dt=dt, lambda_=lambda_, tau=tau,
                     nwarp=nwarp, niter=niter, tol=tol,
                     prefilter=prefilter)

    return coarse_to_fine(I0, I1, solver)
