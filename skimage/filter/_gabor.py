import numpy as np
from scipy import ndimage


__all__ = ['gabor_kernel', 'gabor_filter']


def _sigma_prefactor(bandwidth):
    b = bandwidth
    # See http://www.cs.rug.nl/~imaging/simplecell.html
    return 1.0 / np.pi * np.sqrt(np.log(2)/2.0) * (2.0**b + 1) / (2.0**b - 1)


def gabor_kernel(frequency, theta=0, bandwidth=1, sigma_x=None, sigma_y=None,
                 offset=0):
    """Return complex 2D Gabor filter kernel.

    Frequency and orientation representations of the Gabor filter are similar
    to those of the human visual system. It is especially suitable for texture
    classification using Gabor filter banks.

    Parameters
    ----------
    frequency : float
        Frequency of the harmonic function.
    theta : float
        Orientation in radians. If 0, the harmonic is in the x-direction.
    bandwidth : float
        The bandwidth captured by the filter. For fixed bandwidth, `sigma_x`
        and `sigma_y` will decrease with increasing frequency. This value is
        ignored if `sigma_x` and `sigma_y` are set by the user.
    sigma_x, sigma_y : float
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that `sigma_x` controls the *vertical* direction.
    offset : float, optional
        Phase offset of harmonic function in radians.

    Returns
    -------
    g : complex array
        Complex filter kernel.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Gabor_filter
    .. [2] http://mplab.ucsd.edu/tutorials/gabor.pdf

    """
    if sigma_x is None:
        sigma_x = _sigma_prefactor(bandwidth) / frequency
    if sigma_y is None:
        sigma_y = _sigma_prefactor(bandwidth) / frequency

    n_stds = 3
    x0 = np.ceil(max(np.abs(n_stds * sigma_x * np.cos(theta)),
                     np.abs(n_stds * sigma_y * np.sin(theta)), 1))
    y0 = np.ceil(max(np.abs(n_stds * sigma_y * np.cos(theta)),
                     np.abs(n_stds * sigma_x * np.sin(theta)), 1))
    y, x = np.mgrid[-y0:y0+1, -x0:x0+1]

    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)

    g = np.zeros(y.shape, dtype=np.complex)
    g[:] = np.exp(-0.5 * (rotx**2 / sigma_x**2 + roty**2 / sigma_y**2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= np.exp(1j * (2 * np.pi * frequency * rotx + offset))

    return g


def gabor_filter(image, frequency, theta=0, bandwidth=1, sigma_x=None,
                 sigma_y=None, offset=0, mode='reflect', cval=0):
    """Return real and imaginary responses to Gabor filter.

    The real and imaginary parts of the Gabor filter kernel are applied to the
    image and the response is returned as a pair of arrays.

    Frequency and orientation representations of the Gabor filter are similar
    to those of the human visual system. It is especially suitable for texture
    classification using Gabor filter banks.

    Parameters
    ----------
    image : array
        Input image.
    frequency : float
        Frequency of the harmonic function.
    theta : float
        Orientation in radians. If 0, the harmonic is in the x-direction.
    bandwidth : float
        The bandwidth captured by the filter. For fixed bandwidth, `sigma_x`
        and `sigma_y` will decrease with increasing frequency. This value is
        ignored if `sigma_x` and `sigma_y` are set by the user.
    sigma_x, sigma_y : float
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that `sigma_x` controls the *vertical* direction.
    offset : float, optional
        Phase offset of harmonic function in radians.

    Returns
    -------
    real, imag : arrays
        Filtered images using the real and imaginary parts of the Gabor filter
        kernel.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Gabor_filter
    .. [2] http://mplab.ucsd.edu/tutorials/gabor.pdf

    """

    g = gabor_kernel(frequency, theta, bandwidth, sigma_x, sigma_y, offset)

    filtered_real = ndimage.convolve(image, np.real(g), mode=mode, cval=cval)
    filtered_imag = ndimage.convolve(image, np.imag(g), mode=mode, cval=cval)

    return filtered_real, filtered_imag
