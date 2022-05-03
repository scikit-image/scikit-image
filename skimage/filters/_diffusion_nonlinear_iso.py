import numpy as np
from numba import jit
from .._shared.filters import gaussian
from .._shared.diffusion_utils import (nonlinear_iso_step, aniso_diff_step_AOS,
                                       slice_border, get_diffusivity)


def diffusion_nonlinear_iso(
        image, diffusivity_type='perona-malik', time_step=0.25, num_iters=20,
        scheme='aos', sigma=0.1, alpha=2.):
    """
    Calculates the nonlinear isotropic diffusion of an image.

    Parameters
    ----------
    image : array_like
        Input image.
    time_step : scalar
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
    sigma : scalar
        The standard deviation of the Gaussian filter that is applied to image
        in each diffusion iteration before the gradient estimation.
        Default is 0.1.
    alpha : scalar
        The parameter that determines a treshold contrast for edges.
        Default is 2.0.

    Returns
    -------
    filtered_image : ndarray
        Filtered image

    References
    ----------
    .. [1] Weickert, Joachim. Anisotropic diffusion in image processing.
        Vol. 1. Stuttgart: Teubner, 1998.
    Notes

    Examples
    --------
    Apply a Nonlinear Isotropic Diffusion filter to an image

    >>> from skimage.data import camera
    >>> from skimage.filters._diffusion_nonlinear_iso import diffusion_nonlinear_iso

    >>> filtered_image = diffusion_nonlinear_iso(camera(), time_step=0.25, num_iters=40, scheme='explicit', sigma=0.1, alpha=2.)
    >>> filtered_image2 = diffusion_nonlinear_iso(camera())
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

    border = 1
    type = image.dtype
    img = image.astype(np.float64).copy()
    if len(img.shape) == 3:  # color image
        img = np.pad(img, pad_width=((border, border), (border,
                     border), (0, 0)), mode='edge')  # add Neumann border
        for i in range(img.shape[2]):
            img[:, :, i] = diffusion_nonlinear_iso_grey(
                img[:, :, i], diffusivity_type, time_step, num_iters,
                scheme, sigma, alpha)
    elif len(img.shape) == 2:
        img = np.pad(img, pad_width=border, mode='edge')  # add Neumann border
        img = diffusion_nonlinear_iso_grey(
            img, diffusivity_type, time_step, num_iters,
            scheme, sigma, alpha)

    img = slice_border(img, border)  # remove border
    return img.astype(type)


def diffusion_nonlinear_iso_grey(image, diffusivity_type, time_step, num_iters,
                                 scheme, sigma, alpha):
    if scheme == 'aos':
        diffusion = np.ones(image.shape)
        zeros = np.zeros(image.shape)
    for i in range(num_iters):
        tmp = image.copy()

        gradX, gradY = np.gradient(gaussian(image=tmp, sigma=sigma))

        if scheme == 'explicit':
            nonlinear_iso_step(tmp, image, time_step, gradX,
                               gradY, alpha, diffusivity_type)
        elif scheme == 'aos':
            get_diffusivity_tensor(diffusion, gradX, gradY,
                                   alpha, diffusivity_type)
            aniso_diff_step_AOS(tmp, diffusion,  zeros,
                                diffusion, image, time_step)
        else:
            raise ValueError('invalid scheme')

    return image


@jit(nopython=True)
def get_diffusivity_tensor(out, gradX, gradY, alpha, type):
    for i in range(gradX.shape[0]):
        for j in range(gradX.shape[1]):
            out[i, j] = get_diffusivity(gradX[i, j], gradY[i, j], alpha, type)
