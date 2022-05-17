import numpy as np
from skimage import img_as_float
from ._diffusion_utils import (slice_border, aniso_diff_step_AOS)
from ._diffusion_utils_pythran import (linear_step)


def diffusion_linear(image, time_step=2., num_iters=3, scheme='aos'):
    """
    Calculate the result of linear diffusion equation for an input image
    at time num_iters * time_step.

    Parameters
    ----------
    image : array_like
        Input image.
    time_step : scalar, optional
        Time increment in each diffusion iteration.
        Maximum value for explicit scheme is 0.25, as this is the limit value
        where algorithm is still stable.
        Default is 2.0
    num_iters : scalar, optional
        Number of diffusion iterations.
        Default is 3.
    scheme : {'explicit', 'aos'}, optional
        The computational scheme of the diffusion process.
        'explicit' basic explicit finite difference scheme.
        'aos' stands for additive operator splitting [1].
        Default is 'aos'.

    Returns
    -------
    filtered_image : ndarray
        Filtered image

    Notes
    ----------
    In theory, linear diffusion corresponds to a Gaussian filter with
    sigma = sqrt(2 * time_step * num_iters).
    See skimage.filters.gaussian.
    This function exists for the purpose of consistency with nonlinear
    diffusion filters.

    Time of diffusion is defined as time_step * num_iters. The bigger
    the time_step is, the lower the num_iters parameter has to be
    and the faster the computation is. However, for explicit scheme
    the maximal stable value of time_step is 0.25. If bigger value is
    set by the user, time_step will be automaticaly set to 0.25.

    References
    ----------
    .. [1] Weickert, Joachim. Anisotropic diffusion in image processing.
        Vol. 1. Stuttgart: Teubner, 1998.

    Examples
    --------
    Apply a Linear Diffusion filter to an image

    >>> from skimage.filters.diffusion_linear import diffusion_linear
    >>> from skimage.data import camera
    >>> filtered_image = diffusion_linear(camera(), time_step=0.25, num_iters=40, scheme='explicit')
    >>> filtered_image2 = diffusion_linear(camera())
    """
    if time_step <= 0:
        raise ValueError('invalid time_step')

    if num_iters < 0:
        raise ValueError('invalid num_iters')

    if (len(image.shape) > 3) or (len(image.shape) < 2):
        raise RuntimeError('Unsupported image type')

    scheme = scheme.lower()
    if (scheme == 'explicit') and (time_step > 0.25):
        time_step = 0.25
        print('time_step bigger than 0.25 is unstable for explicit scheme.\
               Time step has been set to 0.25.')

    if scheme not in {"explicit", "aos"}:
        raise ValueError('invalid scheme')

    border = 1
    img = img_as_float(image) * 255  # due to precision error
    #  add Neumann border
    if len(img.shape) == 3:  # color image
        img = np.pad(img, pad_width=((border, border), (border,
                     border), (0, 0)), mode='edge')
        for i in range(img.shape[2]):
            img[:, :, i] = diffusion_linear_grey(
                np.squeeze(img[:, :, i].copy()), time_step, num_iters, scheme)
    else:  # greyscale image
        img = np.pad(img, pad_width=border, mode='edge')
        img = diffusion_linear_grey(
            img, time_step, num_iters, scheme)
    img = slice_border(img, border)  # remove border
    return img / 255


def diffusion_linear_grey(image, time_step, num_iters, scheme):
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

    return image
