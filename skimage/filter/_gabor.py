import numpy as np
from scipy import ndimage


def gabor_kernel(sigma_x, sigma_y, frequency, theta, offset=0):
    """Build complex 2D Gabor filter kernel.

    Frequency and orientation representations of the Gabor filter are similar
    to those of the human visual system. It is especially suitable for texture
    classification using Gabor filter banks.

    Parameters
    ----------
    sigma_x, sigma_y : float
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that `sigma_x` controls the *vertical* direction.
    frequency : float
        Frequency of the harmonic function.
    theta : float
        Orientation in radians.
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


def gabor_filter(image, sigma_x, sigma_y, frequency, theta, offset=0,
                 mode='reflect', cval=0):
    """Perform Gabor filtering.

    The real and imaginary parts of the Gabor filter kernel are applied to the
    image.

    Frequency and orientation representations of the Gabor filter are similar
    to those of the human visual system. It is especially suitable for texture
    classification using Gabor filter banks.

    Parameters
    ----------
    sigma_x, sigma_y : float
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that `sigma_x` controls the *vertical* direction.
    frequency : float
        Frequency of the harmonic function.
    theta : float
        Orientation in radians.
    offset : float, optional
        Phase offset of harmonic function in radians.

    Returns
    -------
    real, imag : complex arrays
        Filtered images using the real and imaginary parts of the Gabor filter
        kernel.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Gabor_filter
    .. [2] http://mplab.ucsd.edu/tutorials/gabor.pdf

    """

    g = gabor_kernel(sigma_x, sigma_y, frequency, theta, offset)

    filtered_real = ndimage.convolve(image, np.real(g), mode=mode, cval=cval)
    filtered_imag = ndimage.convolve(image, np.imag(g), mode=mode, cval=cval)

    return filtered_real, filtered_imag
