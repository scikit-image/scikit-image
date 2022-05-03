import numpy as np
from .._shared.diffusion_utils import (linear_step,
                                       aniso_diff_step_AOS, slice_border)


def diffusion_linear(image, time_step=0.25, num_iters=20, scheme='aos', sigma=2.5, alpha=0.01):
    """
    Calculates the linear diffusion of an image.

    Parameters
    ----------
    image : array_like
        Input image.
    time_step : scalar
        Time increment in each diffusion iteration.
        Default is 0.25.
    num_iters : scalar
        Number of diffusion iterations.
        Default is 20.
    scheme : {'explicit', 'aos'}, optional
        The computational scheme of the diffusion process.
        'explicit'
        'aos' stands for additive operator splitting [1].
        Default is 'aos'.
    sigma : scalar
        The standard deviation of the Gaussian filter that is applied to image
        in each diffusion iteration.
        Default is 2.5.
    alpha : scalar
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

    Examples
    --------
    Apply a Linear Diffusion filter to an image

    >>> from skimage.filters._diffusion_linear import diffusion_linear
    >>> from skimage.data import camera
    >>> filtered_image = diffusion_linear(camera(), time_step=0.25, num_iters=40, scheme='explicit', sigma=5, alpha=0.01)

    >>> filtered_image2 = diffusion_linear(camera())
    """
    if alpha <= 0:
        raise ValueError('invalid alpha')

    if time_step <= 0:
        raise ValueError('invalid time_step')

    if num_iters < 0:
        raise ValueError('invalid num_iters')

    if 2 > len(image.shape) > 3:
        raise RuntimeError('Nonsupported image type')

    border = 1
    type = image.dtype
    img = image.astype(np.float64).copy()
    # add Neumann border
    if len(img.shape) == 3:  # color image
        img = np.pad(img, pad_width=((border, border), (border,
                     border), (0, 0)), mode='edge')
        for i in range(img.shape[2]):
            img[:, :, i] = diffusion_linear_grey(
                img[:, :, i], time_step, num_iters, scheme, sigma, alpha)
    else:
        img = np.pad(img, pad_width=border, mode='edge')
        img = diffusion_linear_grey(
            img, time_step, num_iters, scheme, sigma, alpha)

    img = slice_border(img, border)  # remove border
    return img.astype(type)


def diffusion_linear_grey(image, time_step, num_iters, scheme, sigma, alpha):
    if scheme == 'aos':
        ones = np.ones(image.shape)
        zeros = np.zeros(image.shape)
    for i in range(num_iters):
        tmp = image.copy()
        if scheme == 'aos':
            aniso_diff_step_AOS(tmp, ones, zeros,
                                ones, image, time_step)
        elif scheme == 'explicit':
            linear_step(tmp, image, time_step)
        else:
            raise ValueError('invalid scheme')

    return image
