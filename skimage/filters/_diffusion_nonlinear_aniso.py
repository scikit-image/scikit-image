import numpy as np
from .._shared.filters import gaussian
from .._shared.diffusion_utils import (nonlinear_aniso_step,
                                       aniso_diff_step_AOS, slice_border)
from numba import jit


def diffusion_nonlinear_aniso(image, mode='eed', time_step=0.25, num_iters=20,
                              scheme='aos', sigma_eed=2.5, sigma_ced=0.5, rho=6, alpha=0.01):
    """
    Calculates the nonlinear anisotropic diffusion of an image.
    Namely Edge Enhancing Diffusion[2] and Coherence Enhancing Diffusion [3].

    Parameters
    ----------
    image : array_like
        Input image.
    mode : {'eed', 'ced'}, optional
        'eed' stands for Edge-Enhancing Diffusion.
        'ced' stands for Coherence-Enhancing Diffusion.
        Default is 'eed'.
    time_step : scalar, optional
        Time increment in each diffusion iteration.
        Maximum value for explicit scheme is 0.25, as this is the limit value where algorithm is still stable. 
        Default is 0.25.
    num_iters : scalar
        Number of diffusion iterations.
        Default is 20.
    scheme : {'explicit', 'aos'}, optional
        The computational scheme of the diffusion process.
        'explicit' basic explicit finite difference scheme.
        'aos' stands for additive operator splitting [1].
        Default is 'aos'.
    sigma_eed : scalar, optional
        Used only for Edge Enhancing Diffusion. 
        The standard deviation of the Gaussian filter that is applied to image
        in each diffusion iteration before the gradient estimation.
        Default is 2.5.
    sigma_ced : scalar, optional
        Used only for Coherence Enhancing Diffusion.
        The standard deviation of the Gaussian filter that is applied to image
        in each diffusion iteration before the gradient estimation.
        Default is 0.5.
    rho : scalar, optional
        Used only for Coherence Enhancing Diffusion.
        The standard deviation of the Gaussian filter that is applied
        in order to smooth structure tensor in each diffusion iteration.
        Default is 6.
    alpha : scalar, optional
        The parameter that determines a treshold contrast for edges.
        Default is 0.01.

    Returns
    -------
    filtered_image : ndarray
        Filtered image

    References
    ----------
    .. [1] Weickert, Joachim. Anisotropic diffusion in image processing.
    Vol. 1. Stuttgart: Teubner, 1998.
    Notes
    .. [2] Perona, P., Shiota, T., Malik, J. (1994). Anisotropic Diffusion.
    In: ter Haar Romeny, B.M. (eds) Geometry-Driven Diffusion in Computer Vision.
    Computational Imaging and Vision, vol 1. Springer, Dordrecht.
    .. [3] Weickert, Joachim. "Coherence-enhancing diffusion filtering." 
    International journal of computer vision 31.2 (1999): 111-127.

    Examples
    --------
    Apply a Nonlinear Anisotropic Diffusion filter to an image

    >>> from skimage.data import camera
    >>> from skimage.filters._diffusion_nonlinear_aniso import diffusion_nonlinear_aniso

    Apply Edge Enhancing Diffusion 
    >>> filtered_image_eed = diffusion_nonlinear_aniso(camera(), mode='eed', time_step=0.25, num_iters=40,
                              scheme='explicit', sigma_eed=2.0, sigma_ced=1.0, rho=1.0, alpha=0.01)
    >>> filtered_image_eed_2 = diffusion_nonlinear_aniso(camera())

    Apply Coherence Enhancing Diffusion
    >>> filtered_image_ced = diffusion_nonlinear_aniso(camera(), mode='ced', time_step=0.25, num_iters=40,
                              scheme='aos', sigma_eed=1.0, sigma_ced=0.5, rho=6.0, alpha=0.01)
    >>> filtered_image_ced_2 = diffusion_nonlinear_aniso(camera(), mode='ced')
    """
    if alpha <= 0:
        raise ValueError('invalid alpha')

    if time_step <= 0:
        raise ValueError('invalid time_step')

    if num_iters < 0:
        raise ValueError('invalid num_iters')

    if 2 > len(image.shape) > 3:
        raise RuntimeError('Nonsupported image type')

    if (scheme == 'explicit') and (time_step > 0.25):
        time_step = 0.25
        raise Warning(
            'time_step bigger that 0.25 is unstable for explicit scheme. Time_step has been set to 0.25.')

    type = image.dtype
    img = image.astype(np.float64).copy()
    border = 1
    # add Neumann border
    if len(img.shape) == 3:  # color image
        img = np.pad(img, pad_width=((border, border), (border,
                     border), (0, 0)), mode='edge')
        for i in range(img.shape[2]):
            img[:, :, i] = diffusion_nonlinear_aniso_grey(
                img[:, :, i], mode, time_step, num_iters, scheme, sigma_eed, sigma_ced, rho, alpha, border)
    else:
        img = np.pad(img, pad_width=border, mode='edge')
        img = diffusion_nonlinear_aniso_grey(
            img, mode, time_step, num_iters,
            scheme, sigma_eed, sigma_ced, rho, alpha, border)

    img = slice_border(img, border)
    return img.astype(type)


def diffusion_nonlinear_aniso_grey(image, mode, time_step, num_iters, scheme, sigma_eed, sigma_ced, rho, alpha, border):
    if mode == 'eed':
        image = diffusion_nonlinear_aniso_eed(
            src=image, num_iter=num_iters, tau=time_step, alpha=alpha,
            sig=sigma_eed, scheme=scheme, border=border)
    elif mode == 'ced':
        image = diffusion_nonlinear_aniso_ced(
            src=image, num_iter=num_iters, tau=time_step, alpha=alpha,
            sig=sigma_ced, rho=rho, scheme=scheme, border=border)
    else:
        raise ValueError('invalid mode')
    return image


@jit(nopython=True)
def eed_tensor(Da, Db, Dc, lmbd):
    for i in range(Da.shape[0]):
        for j in range(Da.shape[1]):
            J = np.array([
                [Da[i, j], Db[i, j]],
                [Db[i, j], Dc[i, j]]])

            mi, eigvecs = np.linalg.eig(J)
            mi1 = mi[0]
            mi2 = mi[1]
            ev1 = eigvecs[:, 0]
            ev2 = eigvecs[:, 1]

            if mi2 > mi1:
                mi1, mi2 = mi2, mi1
                ev1, ev2 = ev2, ev1

            # diffusion along edge
            mi2 = 1.0

            # diffusion across edge
            if mi1 <= 2.22045e-16:
                mi1 = 1.0
            else:
                mi1 = 1 - np.exp(-3.31488 / np.power(mi1 / lmbd, 4))

            Da[i, j] = mi1 * ev1[0] * ev1[0] + mi2 * ev2[0] * ev2[0]
            Db[i, j] = mi1 * ev1[0] * ev1[1] + mi2 * ev2[0] * ev2[1]
            Dc[i, j] = mi1 * ev1[1] * ev1[1] + mi2 * ev2[1] * ev2[1]


@jit(nopython=True)
def ced_tensor(Da, Db, Dc, alpha):
    for i in range(Da.shape[0]):
        for j in range(Da.shape[1]):

            J = np.array([
                [Da[i, j], Db[i, j]],
                [Db[i, j], Dc[i, j]]])

            mi, eigvecs = np.linalg.eig(J)
            mi1 = mi[0]
            mi2 = mi[1]
            ev1 = eigvecs[:, 0]
            ev2 = eigvecs[:, 1]

            if mi2 > mi1:
                mi1, mi2 = mi2, mi1
                ev1, ev2 = ev2, ev1

            coherence = np.power(mi1 - mi2, 2)

            # set eigen values
            mi1 = alpha
            if coherence < 2.22045e-16:  # in ref impl coherence is ^2
                mi2 = alpha
            else:
                mi2 = alpha + (1 - alpha) * np.exp(-3.31488 / coherence)

            Da[i, j] = mi1 * ev1[0] * ev1[0] + mi2 * ev2[0] * ev2[0]
            Db[i, j] = mi1 * ev1[0] * ev1[1] + mi2 * ev2[0] * ev2[1]
            Dc[i, j] = mi1 * ev1[1] * ev1[1] + mi2 * ev2[1] * ev2[1]


def diffusion_nonlinear_aniso_ced(src, num_iter, tau, alpha, sig, rho, scheme, border):
    Da = Db = Dc = np.zeros(src.shape).astype(np.float64)

    for i in range(num_iter):
        tmp = src.copy()
        (gradX, gradY) = np.gradient(
            gaussian(image=src, sigma=sig))

        Da = gaussian(image=np.multiply(gradX, gradX),
                      sigma=rho)
        Db = gaussian(image=np.multiply(gradX, gradY),
                      sigma=rho)
        Dc = gaussian(image=np.multiply(gradY, gradY),
                      sigma=rho)

        ced_tensor(Dc, Db, Da, alpha)
        if scheme == 'aos':
            aniso_diff_step_AOS(tmp, Dc,  Db, Da, src, tau)
        elif scheme == 'explicit':
            nonlinear_aniso_step(tmp, src, Da, Db, Dc, tau, border)
        else:
            raise ValueError('invalid scheme')
    return src


def diffusion_nonlinear_aniso_eed(src, num_iter, tau, alpha, sig, scheme, border):
    Da = Db = Dc = np.zeros(src.shape).astype(np.float64)
    for i in range(num_iter):
        tmp = src.copy()

        gradX, gradY = np.gradient(
            gaussian(image=src, sigma=sig).astype(np.float64))
        Da = np.multiply(gradX, gradX).astype(np.float64)
        Db = np.multiply(gradX, gradY).astype(np.float64)
        Dc = np.multiply(gradY, gradY).astype(np.float64)

        eed_tensor(Dc, Db, Da, alpha)
        if scheme == 'aos':
            aniso_diff_step_AOS(tmp, Dc, Db, Da, src, tau)
        elif scheme == 'explicit':
            nonlinear_aniso_step(tmp, src, Da, Db, Dc, tau, border)
        else:
            raise ValueError('invalid scheme')
    return src
