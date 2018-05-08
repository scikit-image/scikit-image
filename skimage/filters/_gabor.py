import collections as coll
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


def _decompose_quasipolar_coords(r, thetas):
    """Decomposes quasipolar coordinates into their cartesian components.

    Parameters
    ----------
    r : float
        Radial coordinate.
    thetas : (N, 1) array
        Quasipolar angles.

    Returns
    -------
    coords : (``N + 1``, 1) array
        Cartesian components of the quasipolar coordinates.

    References
    ----------
    .. [1] Tan Mai Nguyen. N-Dimensional Quasipolar Coordinates - Theory and
           Application. University of Nevada: Las Vegas, Nevada, 2014.
           https://digitalscholarship.unlv.edu/thesesdissertations/2125

    Notes
    -----
    Components ``0``, ``1``, ``...``, ``n`` of the quasipolar coordinates
    correspond to dimensions ``0``, ``1``, ``...``, ``n`` or ``M``, ``...``,
    ``N``, ``P``.

    For a standard ``xy``-coordinate plane, components ``1`` and ``0``
    correspond to ``x`` and ``y``, respectively.

    For a standard ``xyz``-coordinate plane, components ``1``, ``0``, and ``2``
    correspond to ``x``, ``y``, and ``z``, respectively.
    """
    axes = np.size(thetas) + 1
    coords = r * np.ones(axes)

    for which_theta, theta in enumerate(thetas[::-1]):
        sine = np.sin(theta)
        theta_index = axes - which_theta - 1

        for axis in range(theta_index):
            coords[axis] *= sine

        coords[theta_index] *= np.cos(theta)

    return coords


def _gaussian(image, center=0, sigma=1, order=2):
    """Generates gaussian envelope.

    Parameters
    ----------
    image : non-complex array
        Image to seed the filter with.
    center : scalar or vector, optional
        Coordinates to center the image with. Defaults to 0.
    sigma : scalar or vector, optional
        Spatial dimensions of the envelope. Defaults to 1.
    order : int, optional
        Order of the envelope to create. Defaults to 2.

    Returns
    -------
    gauss : (``order``, ``order``)

    References
    ----------
    .. [1] Do CB. 2008. The Multivariate Gaussian Distribution. Stanford
           University (CS 229): Stanford, CA.
           http://cs229.stanford.edu/section/gaussians.pdf
    """
    sigma_prod = np.prod(sigma)

    # normalization factor
    norm = (2 * np.pi) ** (ndim / 2)
    norm *= sigma_prod

    # center image
    image = image - center

    # gaussian envelope
    gauss = np.exp(-0.5 * np.dot(image, image) / sigma_prod ** 2)

    return gauss / norm


def _rotation(src_axis, dst_axis):
    """Generates an matrix for the nD rotation of one axis to face another.

    Parameters
    ----------
    src_axis : (N,) matrix
        Vector representation of the axis that will be rotated.

    dst_axis : (N,) matrix
        Vector representation of the axis to rotate to.

    Returns
    -------
    M : (N, N) array
        Matrix that rotates ``src_axis`` to coincide with ``dst_axis``.

    References
    ----------
    .. [1] Ognyan Ivanov Zhelezov. One Modification which Increases Performance
           of N-Dimensional Rotation Matrix Generation Algorithm. International
           Journal of Chemistry, Mathematics, and Physics, Vol. 2 No. 2, 2018:
           pp. 13-18. https://dx.doi.org/10.22161/ijcmp.2.2.1

    Examples
    --------
    >>> X = np.asarray([1, 0])
    >>> Y = np.asarray([0.5, 0.5])

    >>> M = _rotation(X, Y)
    >>> Z = np.matmul(M, Y[..., np.newaxis])

    >>> np.allclose(Z, Y[np.newaxis])
    True
    """
    X = np.array(src_axis)
    Y = np.array(dst_axis)


    def rot(axis, diffs):
        ndim = len(axis)
        num_diffs = len(diffs)

        x = axis
        w = diffs

        R = np.eye(ndim)  # Initial rotation matrix = Identity matrix

        step = 1  # Initial step
        while step < ndim:  # Loop to create matrices of stages
            A = np.eye(ndim)

            n = 0
            while n < ndim - step and n + step < num_diffs:
                r2 = x[w[n]] * x[w[n]] + x[w[n + step]] * x[w[n + step]]
                if r2 > 0:
                    r = np.sqrt(r2)
                    pcos = x[w[n]] / r  # Calculation of coefficients
                    psin = -x[w[n + step]] / r
                    # Base 2-dimensional rotation
                    A[w[n], w[n]] = pcos
                    A[w[n], w[n + step]] = -psin
                    A[w[n + step], w[n]] = psin
                    A[w[n + step], w[n + step]] = pcos
                    x[w[n + step]] = 0
                    x[w[n]] = r

                n += step << 1  # Move to the next base operation

            step <<= 1  # multiply by 2
            R = np.matmul(A, R)  # Multiply R by current matrix of stage A

        return R


    normX = np.linalg.norm(X)
    normY = np.linalg.norm(Y)

    if not np.isclose(normX, normY):           # Set norm of Y equal to norm
        Y = (normX / normY) * Y  # of X if they are different

    w = np.nonzero(~np.isclose(X, Y))[0]  # indices of difference

    Mx = rot(X, w)
    My = rot(Y, w)
    M = np.matmul(My.T, Mx)

    return M


def gabor_kernel(frequency, theta=0, bandwidth=1, sigma=None,
                 sigma_y=None, n_stds=3, offset=None, ndim=2, **kwargs):
    """Return complex nD Gabor filter kernel.

    A gabor kernel is a Gaussian kernel modulated by a complex harmonic function.
    Harmonic function consists of an imaginary sine function and a real
    cosine function. Spatial frequency is inversely proportional to the
    wavelength of the harmonic and to the standard deviation of a Gaussian
    kernel. The bandwidth is also inversely proportional to the standard
    deviation.

    Parameters
    ----------
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    theta : float or array of floats, optional
        Orientation in radians. If 0, the harmonic is in the x-direction.
    bandwidth : float, optional
        The bandwidth captured by the filter. For fixed bandwidth, `sigma`
        will decrease with increasing frequency. This value is ignored if
        `sigma` are set by the user.
    sigma : float or array of floats, optional
        Standard deviation. These directions apply to the kernel *before*
        rotation.
    n_stds : scalar, optional
        The linear size of the kernel is n_stds (3 by default) standard
        deviations
    offset : float, optional
        Phase offset of harmonic function in radians.
    ndim : int, optional
        Dimensionality of the kernel. Defaults to 2.

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

    default_sigma = _sigma_prefactor(bandwidth) / frequency

    # handle translation
    if not isinstance(theta, coll.Iterable):
        theta = (0,) * (ndim - 1)

    if not isinstance(sigma, coll.Iterable):
        sigma = np.array([sigma] * ndim)
    else:
        sigma = np.append(sigma, [None] * (ndim - len(sigma)))
    sigma[(sigma == None).nonzero()] = default_sigma  # noqa
    sigma = sigma.astype(None)

    coords = _decompose_quasipolar_coords(frequency, theta)
    base_axis = np.zeros(ndim)
    base_axis[0] = 1
    rot = _rotation(base_axis, coords)

    x = ...
    rotx = np.matmul(rot, x)

    gauss = _gaussian(rotx, center=0, sigma=sigma, ndim=ndim)

    compx = np.matmul(x, coords)

    # complex harmonic function
    harmonic = np.exp(1j * (2 * np.pi * compx.sum() + offset))

    g = norm * np.matmul(gauss, harmonic)

    return g.view(cls)


def gabor(image, frequency=None, theta=0, bandwidth=1, sigma=None, sigma_y=None,
          n_stds=3, offset=None, mode='reflect', cval=0, kernel=None, **kwargs):
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
    image : array_like
        Input image.
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    theta : float or array of floats, optional
        Orientation in radians. If 0, the harmonic is in the x-direction.
    bandwidth : float, optional
        The bandwidth captured by the filter. For fixed bandwidth, `sigma`
        will decrease with increasing frequency. This value is ignored if
        `sigma` are set by the user.
    sigma : float or array of floats, optional
        Standard deviation. These directions apply to the kernel *before*
        rotation.
    n_stds : scalar, optional
        The linear size of the kernel is n_stds (3 by default) standard
        deviations
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
    if kernel is None:
        if frequency is None:
            raise TypeError("gabor() must specify 'frequency' "
                            "if 'kernel' is not provided")
        kernel = gabor_kernel(frequency, theta, bandwidth, sigma, sigma_y,
                              n_stds, offset, ndim=image.ndim, **kwargs)
    else:
        if frequency is not None:
            warn("gabor() received arguments of "
                 "both 'kernel' and 'frequency'; "
                 "'frequency' will be ignored")
        assert_nD(np.ndim(image), kernel.ndim)

    filtered_real = ndi.convolve(image, np.real(kernel), mode=mode, cval=cval)
    filtered_imag = ndi.convolve(image, np.imag(kernel), mode=mode, cval=cval)

    return filtered_real, filtered_imag
