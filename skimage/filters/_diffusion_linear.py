import numpy as np
from .._shared.diffusion_utils import (linear_step,
                                       aniso_diff_step_AOS, slice_border)
from skimage import data, img_as_float


def diffusion_linear(image, time_step=2., num_iters=3, scheme='aos'):
    """
    Calculates the result of linear diffusion equation for an input image
    at time num_iters * time_step.
    In theory, it corresponds to a Gaussian filter with
    sigma = sqrt(2 * time_step * num_iters).
    See skimage.filters.gaussian. This function exists for the purpose
    of consistency.
    with nonlinear diffusion filters.
  
    Parameters
    ----------
    image : array_like
        Input image.
    time_step : scalar
        Time increment in each diffusion iteration.
        Maximum value for explicit scheme is 0.25, as this is the limit value
        where algorithm is still stable.
        Default is 0.25.
    num_iters : scalar
        Number of diffusion iterations.
        Default is 20.
    scheme : {'explicit', 'aos'}, optional
        The computational scheme of the diffusion process.
        'explicit' basic explicit finite difference scheme.
        'aos' stands for additive operator splitting [1].
        Default is 'aos'.
    alpha : scalar
        The parameter that determines a treshold contrast for edges.
        Default is 0.01.

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

    Examples
    --------
    Apply a Linear Diffusion filter to an image

    >>> from skimage.filters._diffusion_linear import diffusion_linear
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

    if (scheme == 'explicit') and (time_step > 0.25):
        time_step = 0.25
        print('time_step bigger than 0.25 is unstable for explicit scheme.\
               Time step has been set to 0.25.')

    if (scheme != 'explicit') and (scheme != 'aos'):
        raise ValueError('invalid scheme')

    border = 1
    type = image.dtype
    #img = image.astype(np.float64).copy()
    img = img_as_float(image)*255 #  due to precision error
    # add Neumann border
    if len(img.shape) == 3:  # color image
        img = np.pad(img, pad_width=((border, border), (border,
                     border), (0, 0)), mode='edge')
        for i in range(img.shape[2]):
            img[:, :, i] = diffusion_linear_grey(
                img[:, :, i], time_step, num_iters, scheme)
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
