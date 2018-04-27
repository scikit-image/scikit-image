import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import assert_nD
from warnings import warn


__all__ = ['gabor_kernel', 'gabor']


def _sigma_prefactor(bandwidth):
    b = bandwidth
    # See http://www.cs.rug.nl/~imaging/simplecell.html
    return 1.0 / np.pi * np.sqrt(np.log(2) / 2.0) * \
        (2.0 ** b + 1) / (2.0 ** b - 1)


def _get_quasipolar_components(r, *thetas):
    """Nguyen, Tan Mai, "N-Dimensional Quasipolar Coordinates - Theory and Application" (2014). UNLV Theses, Dissertations, Professional Papers, and Capstones. 2125. https://digitalscholarship.unlv.edu/thesesdissertations/2125
    """
    axes = len(thetas) + 1
    coords = r * np.ones(axes)

    for which_theta, theta in enumerate(thetas[::-1]):
        sine = np.sin(theta)
        theta_index = axes - which_theta - 1

        for axis in range(theta_index):
            coords[axis] *= sine

        coords[theta_index] *= np.cos(theta)

    return coords


def _gaussian(image, center=None, sigma=1, ndim=2):
    """Do CB. 2008. The Multivariate Gaussian Distribution. Stanford University (CS 229): Stanford, CA. http://cs229.stanford.edu/section/gaussians.pdf"""
    # normalization factor
    norm = (2 * np.pi) ** (ndim / 2)
    norm *= np.prod(sigma)

    # center image
    centered_image = image - center

    im = centered_image / sigma

    # gaussian envelope
    gauss = np.exp(-0.5 * np.dot(im, im))

    return gauss / norm


class gabor_kernel(np.ndarray):
    """Return complex nD Gabor filter kernel.

    Gabor kernel is a Gaussian kernel modulated by a complex harmonic function.
    Harmonic function consists of an imaginary sine function and a real
    cosine function. Spatial frequency is inversely proportional to the
    wavelength of the harmonic and to the standard deviation of a Gaussian
    kernel. The bandwidth is also inversely proportional to the standard
    deviation.

    Parameters
    ----------
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    theta : float, optional
        Orientation in radians. If 0, the harmonic is in the x-direction.
    bandwidth : float, optional
        The bandwidth captured by the filter. For fixed bandwidth, `sigma_x`
        and `sigma_y` will decrease with increasing frequency. This value is
        ignored if `sigma_x` and `sigma_y` are set by the user.
    sigma_x, sigma_y : float, optional
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that `sigma_x` controls the *vertical* direction.
    n_stds : scalar, optional
        The linear size of the kernel is n_stds (3 by default) standard
        deviations
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

    Examples
    --------
    >>> from skimage.filters import gabor_kernel
    >>> from skimage import io
    >>> from matplotlib import pyplot as plt  # doctest: +SKIP

    >>> gk = gabor_kernel(frequency=0.2)
    >>> plt.figure()        # doctest: +SKIP
    >>> io.imshow(gk.real)  # doctest: +SKIP
    >>> io.show()           # doctest: +SKIP

    >>> # more ripples (equivalent to increasing the size of the
    >>> # Gaussian spread)
    >>> gk = gabor_kernel(frequency=0.2, bandwidth=0.1)
    >>> plt.figure()        # doctest: +SKIP
    >>> io.imshow(gk.real)  # doctest: +SKIP
    >>> io.show()           # doctest: +SKIP
    """
    def __new__(cls, frequency, theta=None, bandwidth=1, sigma=None,
                sigma_y=None, n_stds=3, offset=None, ndim=2, **kwargs):
        # Import has to be here due to circular import error
        from ..transform import rotate

        # handle deprecation
        message = ('Using deprecated, 2D-only interface to gabor_kernel. '
                   'This interface will be removed in scikit-image 0.16. Use '
                   'gabor_kernel(frequency, sigma=(sigma_y, sigma_x)).')

        if sigma_y is not None:
            warn(message)
            if 'sigma_x' in kwargs:
                sigma = (sigma_y, kwargs['sigma_x'])
            else:
                sigma = (sigma_y, sigma)

        # handle translation
        if theta is None:
            theta = (0,) * (ndim - 1)
        elif type(offset) is tuple:
            theta += (0,) * (len(theta) - (ndim - 1))
        else:
            theta = (theta,) * (ndim - 1)

        if type(sigma) is tuple:
            sigma += (None,) * (ndim - len(sigma))
            sigma = np.asarray(sigma)
        else:
            sigma = np.array([sigma] * ndim)
        sigma[(sigma == None).nonzero()] = (_sigma_prefactor(bandwidth)  # noqa
                                            / frequency)
        sigma = sigma.astype(None)

        x = ...

        gauss = _gaussian(x, center=0, sigma=sigma, ndim=ndim)

        rotx = np.matmul(x, _get_quasipolar_components(frequency, **theta))

        # complex harmonic function
        harmonic = np.exp(1j * (2 * np.pi * rotx.sum() + offset))

        g = norm * np.matmul(gauss, harmonic)

        return g.view(cls)

    def apply(self, image, mode='reflect', cval=0):
        """Return real and imaginary responses to Gabor filter.

        The real and imaginary parts of the Gabor filter kernel are applied to
        the image and the response is returned as a pair of arrays.

        Gabor filter is a linear filter with a Gaussian kernel which is
        modulated by a sinusoidal plane wave. Frequency and orientation
        representations of the Gabor filter are similar to those of the human
        visual system. Gabor filter banks are commonly used in computer vision
        and image processing. They are especially suitable for edge detection
        and texture classification.

        Parameters
        ----------
        image : array where image.ndim == self.ndim
            Input image.
        mode : {'constant', 'nearest', 'reflect', 'mirror', 'wrap'}, optional
            Mode used to convolve image with a kernel, passed to `ndi.convolve`
        cval : scalar, optional
            Value to fill past edges of input if `mode` of convolution is
            'constant'. The parameter is passed to `ndi.convolve`.

        Returns
        -------
        real, imag : arrays
            Filtered images using the real and imaginary parts of the Gabor
            filter kernel. Images are of the same dimensions as the input one.

        References
        ----------
        .. [1] http://en.wikipedia.org/wiki/Gabor_filter
        .. [2] http://mplab.ucsd.edu/tutorials/gabor.pdf

        Examples
        --------
        >>> from skimage.filters import gabor_kernel
        >>> from skimage import data, io
        >>> from matplotlib import pyplot as plt  # doctest: +SKIP

        >>> image = data.coins()
        >>> # detecting edges in a coin image
        >>> filt_real, filt_imag = gabor_kernel(frequency=0.6).apply(image)
        >>> plt.figure()            # doctest: +SKIP
        >>> io.imshow(filt_real)    # doctest: +SKIP
        >>> io.show()               # doctest: +SKIP

        >>> # less sensitivity to finer details with the lower frequency kernel
        >>> filt_real, filt_imag = gabor_kernel(frequency=0.1).apply(image)
        >>> plt.figure()            # doctest: +SKIP
        >>> io.imshow(filt_real)    # doctest: +SKIP
        >>> io.show()               # doctest: +SKIP
        """
        assert_nD(image, self.ndim)

        filtered_real = ndi.convolve(image, self.real, mode=mode, cval=cval)
        filtered_imag = ndi.convolve(image, self.imag, mode=mode, cval=cval)

        return filtered_real, filtered_imag


def gabor(image, *args, mode='reflect', cval=0, **kwargs):
    """Return real and imaginary responses to Gabor filter.

    The real and imaginary parts of the Gabor filter kernel are applied to the
    image and the response is returned as a pair of arrays.

    Gabor filter is a linear filter with a Gaussian kernel which is modulated
    by a sinusoidal plane wave. Frequency and orientation representations of
    the Gabor filter are similar to those of the human visual system.
    Gabor filter banks are commonly used in computer vision and image
    processing. They are especially suitable for edge detection and texture
    classification.

    Parameters
    ----------
    image : array where image.ndim == ndim
        Input image.
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    theta : float, optional
        Orientation in radians. If 0, the harmonic is in the x-direction.
    bandwidth : float, optional
        The bandwidth captured by the filter. For fixed bandwidth, `sigma_x`
        and `sigma_y` will decrease with increasing frequency. This value is
        ignored if `sigma_x` and `sigma_y` are set by the user.
    sigma_x, sigma_y : float, optional
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that `sigma_x` controls the *vertical* direction.
    n_stds : scalar, optional
        The linear size of the kernel is n_stds (3 by default) standard
        deviations.
    offset : float, optional
        Phase offset of harmonic function in radians.
    mode : {'constant', 'nearest', 'reflect', 'mirror', 'wrap'}, optional
        Mode used to convolve image with a kernel, passed to `ndi.convolve`
    cval : scalar, optional
        Value to fill past edges of input if `mode` of convolution is
        'constant'. The parameter is passed to `ndi.convolve`.

    Returns
    -------
    real, imag : arrays
        Filtered images using the real and imaginary parts of the Gabor filter
        kernel. Images are of the same dimensions as the input one.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Gabor_filter
    .. [2] http://mplab.ucsd.edu/tutorials/gabor.pdf

    Examples
    --------
    >>> from skimage.filters import gabor
    >>> from skimage import data, io
    >>> from matplotlib import pyplot as plt  # doctest: +SKIP

    >>> image = data.coins()
    >>> # detecting edges in a coin image
    >>> filt_real, filt_imag = gabor(image, frequency=0.6)
    >>> plt.figure()            # doctest: +SKIP
    >>> io.imshow(filt_real)    # doctest: +SKIP
    >>> io.show()               # doctest: +SKIP

    >>> # less sensitivity to finer details with the lower frequency kernel
    >>> filt_real, filt_imag = gabor(image, frequency=0.1)
    >>> plt.figure()            # doctest: +SKIP
    >>> io.imshow(filt_real)    # doctest: +SKIP
    >>> io.show()               # doctest: +SKIP
    """
    g = gabor_kernel(*args, **kwargs)

    return g.apply(image, mode=mode, cval=cval)
