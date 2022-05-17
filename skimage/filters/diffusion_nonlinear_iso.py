import numpy as np
from .._shared.filters import gaussian
from ._diffusion_utils import get_diffusivity
from skimage import img_as_float64
from ._diffusion_utils import slice_border
from ._diffusion_utils_pythran import nonlinear_iso_step
from ._diffusion_utils import aniso_diff_step_AOS


def diffusion_nonlinear_iso(
        image, diffusivity_type='perona-malik', time_step=1., num_iters=20,
        scheme='aos', sigma=1.0, lmbd=2.):
    """
    Calculate the nonlinear isotropic diffusion of an image.

    Parameters
    ----------
    image : array_like
        Input image.
    time_step : scalar, optional
        Time increment in each diffusion iteration.
        Maximum value for explicit scheme is 0.25, as this is the limit
        value where algorithm is still stable.
        Default is 1.0.
    diffusivity_type : {'perona-malik', 'charbonnier', 'exponential'}, optional
        Type of diffusivity. The diffusivity term in the diffusion equation
        is set according to the chosen diffusivity type.
        Default is 'perona-malik'
    num_iters : scalar, optional
        Number of diffusion iterations.
        Default is 20.
    scheme : {'explicit', 'aos'}, optional
        The computational scheme of the diffusion process.
        'explicit' basic explicit finite difference scheme.
        'aos' stands for additive operator splitting [1].
        Default is 'aos'.
    sigma : scalar, optional
        The standard deviation of the Gaussian filter that is applied to image
        in each diffusion iteration before the gradient estimation.
        Default is 1.0.
    lmbd : scalar, optional
        Lambda parameter that determines a treshold contrast for edges.
        Default is 2.0.

    Returns
    -------
    filtered_image : ndarray
        Filtered image

    Notes
    ----------
    Time of diffusion is defined as time_step * num_iters. The bigger
    the time_step is, the lower the num_iters parameter has to be
    and the faster the computation is. However, for explicit scheme
    the maximal stable value of time_step is 0.25. If bigger value is
    set by the user, time_step will be automaticaly set to 0.25.

    References
    ----------
    .. [1] Weickert, Joachim. Anisotropic diffusion in image processing.
        Vol. 1. Stuttgart: Teubner, 1998.
    Notes

    Examples
    --------
    Apply a Nonlinear Isotropic Diffusion filter to an image

    >>> from skimage.data import camera
    >>> from skimage.filters.diffusion_nonlinear_iso import diffusion_nonlinear_iso
    >>> filtered_image = diffusion_nonlinear_iso(camera(), time_step=0.25, num_iters=40, scheme='explicit', sigma=0.1, lmbd=2.)
    >>> filtered_image2 = diffusion_nonlinear_iso(camera())
    """

    if lmbd <= 0:
        raise ValueError('invalid lambda parameter.')

    if time_step <= 0:
        raise ValueError('invalid time_step.')

    if num_iters < 0:
        raise ValueError('invalid num_iters.')

    if (len(image.shape) > 3) or (len(image.shape) < 2):
        raise RuntimeError('Nonsupported image type.')

    scheme = scheme.lower()
    if (scheme == 'explicit') and (time_step > 0.25):
        time_step = 0.25
        print('time_step bigger than 0.25 is unstable for explicit scheme.\
               Time step has been set to 0.25.')

    if scheme not in {"explicit", "aos"}:
        raise ValueError('invalid scheme')

    border = 1
    img = img_as_float64(image) * 255  # due to precision error
    if len(img.shape) == 3:  # color image
        img = np.pad(img, pad_width=((border, border), (border,
                     border), (0, 0)), mode='edge')  # add Neumann border
        for i in range(img.shape[2]):
            img[:, :, i] = diffusion_nonlinear_iso_grey(
                np.squeeze(img[:, :, i].copy()), diffusivity_type, time_step,
                num_iters, scheme, sigma, lmbd)
    elif len(img.shape) == 2:
        img = np.pad(img, pad_width=border, mode='edge')  # add Neumann border
        img = diffusion_nonlinear_iso_grey(
            img, diffusivity_type, time_step, num_iters,
            scheme, sigma, lmbd)

    img = slice_border(img, border)  # remove border
    return img / 255


def diffusion_nonlinear_iso_grey(image, diffusivity_type, time_step, num_iters,
                                 scheme, sigma, lmbd):
    if scheme == 'aos':
        diffusion = img_as_float64(np.ones(image.shape))
        zeros = img_as_float64(np.zeros(image.shape))
    for i in range(num_iters):
        tmp = image.copy()

        gradX, gradY = np.gradient(gaussian(image=tmp, sigma=sigma))

        if scheme == 'explicit':
            nonlinear_iso_step(tmp, image, time_step, gradX,
                               gradY, lmbd, diffusivity_type)
        elif scheme == 'aos':
            get_diffusivity_tensor(diffusion, gradX, gradY,
                                   lmbd, diffusivity_type)
            aniso_diff_step_AOS(tmp, diffusion, zeros,
                                diffusion, image, time_step)
    return image


def get_diffusivity_tensor(out, gradX, gradY, lmbd, type):
    for i in range(gradX.shape[0]):
        for j in range(gradX.shape[1]):
            out[i, j] = get_diffusivity(gradX[i, j], gradY[i, j], lmbd, type)
