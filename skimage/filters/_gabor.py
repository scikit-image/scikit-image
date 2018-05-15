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


def _decompose_quasipolar_coords(r, thetas, axes=0):
    """Decomposes quasipolar coordinates into their cartesian components.

    Parameters
    ----------
    r : float
        Radial coordinate.
    thetas : (N, ) array
        Quasipolar angles.
    axes : int or sequence of int
        Ordering of axes that defines the plane with regards to
        orientation. Ordering not specified will be padded with
        remaining axes in ascending order. Non-iterable values
        will be treated as the single element of a tuple.
        For classical cartesian ordering `(x, y, ...)`, set to `1`.

    Returns
    -------
    coords : (``N + 1``, ) array
        Cartesian components of the quasipolar coordinates.

    References
    ----------
    .. [1] Tan Mai Nguyen. N-Dimensional Quasipolar Coordinates - Theory and
           Application. University of Nevada: Las Vegas, Nevada, 2014.
           https://digitalscholarship.unlv.edu/thesesdissertations/2125

    Notes
    -----
    Quasipolar coordinate decomposition is defined as follows:

    .. math::

         \left\{
         \begin{array}{llllll}
	         x_0     & \quad = r \sin \theta_0 \sin \theta_1 ... \sin \theta_{n-1} \\
	         x_1     & \quad = r \cos \theta_0 \sin \theta_1 ... \sin \theta_{n-1} \\
	         x_2     & \quad = r \cos \theta_1 \sin \theta_2 ... \sin \theta_{n-1} \\
	         ...                                                                   \\
	         x_{n-1} & \quad = r \cos \theta_{n-2} \sin \theta_{n-1}               \\
	         x_n     & \quad = r \cos \theta_{n-1}
         \end{array}
         \right.

    For polar coordinates:

    .. math::

         \left\{
         \begin{array}{ll}
	         y = x_0 = r \sin \theta_0 \\
	         x = x_1 = r \cos \theta_0
         \end{array}
         \right.

    For spherical coordinates:

    .. math::

         \left\{
         \begin{array}{lll}
	         y = x_0 = r \sin \theta_0 \sin \theta_1 \\
	         x = x_1 = r \cos \theta_0 \sin \theta_1 \\
	         z = x_2 = r \cos \theta_1
         \end{array}
         \right.

    Examples
    --------
    >>> _decompose_quasipolar_coords(1, (0))
    [ 0., 1.]
    >>> _decompose_quasipolar_coords(1, (0), leading_axis=1)
    [ 1., 0.]
    >>> _decompose_quasipolar_coords(10, (np.pi / 2, 0))
    [ 10., 0., 0.]
    >>> _decompose_quasipolar_coords(5, (np.pi / 2, 0), leading_axis=2)
    [ 0., 5., 0.]
    """
    num_axes = len(thetas) + 1
    coords = r * np.ones(num_axes)

    if not isinstance(axes, coll.Iterable):
        axes = (axes,)
    axes = np.append(axes, np.setdiff1d(range(num_axes), axes))

    for which_theta, theta in enumerate(thetas[::-1]):
        sine = np.sin(theta)
        theta_index = num_axes - which_theta - 1

        for axis in range(theta_index):
            coords[axis] *= sine

        coords[theta_index] *= np.cos(theta)

    return coords[axes]


def _gaussian_kernel(image, center=0, sigma=1):
    """Multi-dimensional Gaussian kernel.

    Parameters
    ----------
    image : non-complex array
        Linear space to map the filter to.
    center : scalar or sequence of scalars, optional
        Center of Gaussian kernel. The coordinates of the center are given for
        each axis as a sequence, or as a single number, in which case it is
        equal for all axes.
    sigma : scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel. The standard deviations of the
        Gaussian filter are given for each axis as a sequence, or as a single
        number, in which case it is equal for all axes.

    Returns
    -------
    gauss : (``image.ndim``, ``image.ndim``)
        Filter kernel.

    References
    ----------
    .. [1] Bart M. ter Haar Romeny. Front-End Vision and Multi-Scale Image
           Analysis. Computation Imaging and Vision, Vol. 27, Springer:
           Dordrecht, 2003: pp. 37-51.
           https://doi.org/10.1007/978-1-4020-8840-7_3
    """
    sigma_prod = np.prod(sigma)

    # normalization factor
    norm = (2 * np.pi) ** (image.ndim / 2) * sigma_prod

    # center image
    image = image - center

    # gaussian function
    gauss = np.exp(-0.5 * np.sum(image ** 2 / sigma_prod ** 2, axis=0))

    return gauss / norm


def _normalize(x):
    """Normalizes an array.

    Parameters
    ----------
    x : array_like
        Array to normalize.

    Returns
    -------
    u : array
        Unitary array.

    Examples
    --------
    >>> x = np.arange(5)
    >>> uX = _normalize(x)
    >>> np.isclose(np.lingalg.norm(uX), 1)
    True
    """
    u = np.asarray(x)

    norm = np.linalg.norm(u)

    if not np.isclose(norm, 1):
        u = u / norm

    return u


def _compute_projection_matrix(axis, indices=None):
    """Generates a matrix that projects an axis onto the 0th coordinate axis.

    Parameters
    ----------
    axis : (N, ) array
        Unit vector.
    indices : sequence of int
        Indices of the components of `axis` that should be transformed.
        If `None`, defaults to all of the indices of `axis`.

    Returns
    -------
    R : (N, N) array
        Orthogonal projection matrix.

    References
    ----------
    .. [1] Ognyan Ivanov Zhelezov. One Modification which Increases Performance
           of N-Dimensional Rotation Matrix Generation Algorithm. International
           Journal of Chemistry, Mathematics, and Physics, Vol. 2 No. 2, 2018:
           pp. 13-18. https://dx.doi.org/10.22161/ijcmp.2.2.1
    """
    ndim = len(axis)

    if indices is None:
        indices = range(ndim)

    x = axis
    w = indices

    R = np.eye(ndim)  # Initial rotation matrix = Identity matrix

    # Loop to create matrices of stages
    for step in np.round(2 ** np.arange(np.log2(ndim))).astype(int):
        A = np.eye(ndim)

        for n in range(0, ndim - step, step * 2):
            if n + step >= len(w):
                break

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

        R = np.matmul(A, R)  # Multiply R by current matrix of stage A

    return R


def _compute_rotation_matrix(src, dst, use_homogeneous_coordinates=False):
    """Generates a matrix for the rotation of one vector to the direction
    of another.

    Parameters
    ----------
    src : (N, ) array
        Vector to rotate.
    dst : (N, ) array
        Vector of desired direction.
    use_homogeneous_coordinates : bool
        If the input vectors should be treated as homoegeneous coordinates.

    Returns
    -------
    M : (N, N) array
        Matrix that rotates `src` to coincide with `dst`.

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
    >>> Z = np.matmul(M, X)

    >>> np.allclose(Z, Y)
    True
    """
    X = _normalize(np.array(src))
    Y = _normalize(np.array(dst))

    if use_homogeneous_coordinates:
        X[-1] = 1
        Y[-1] = 1

    w = np.flatnonzero(~np.isclose(X, Y))  # indices of difference

    Mx = _compute_projection_matrix(X, w)
    My = _compute_projection_matrix(Y, w)
    M = np.matmul(My.T, Mx)

    return M


def gabor_kernel(frequency, theta=0, bandwidth=1, sigma=None, sigma_y=None,
                 n_stds=3, offset=0, axes=1, ndim=2, **kwargs):
    """Multi-dimensional complex Gabor kernel.

    A Gabor kernel is a Gaussian kernel modulated by a complex harmonic function.
    Harmonic function consists of an imaginary sine function and a real
    cosine function. Spatial frequency is inversely proportional to the
    wavelength of the harmonic and to the standard deviation of a Gaussian
    kernel. The bandwidth is also inversely proportional to the standard
    deviation.

    Parameters
    ----------
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    theta : float or sequence of floats, optional
        Orientation in radians. If 0, the harmonic is in the x-direction.
    theta : scalar or sequence of scalars, optional
        Orientation in radians. The angles that describe the orientation
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    bandwidth : float, optional
        The bandwidth captured by the filter. For fixed bandwidth, `sigma`
        will decrease with increasing frequency. This value is ignored if
        `sigma` are set by the user.
    sigma : scalar or sequence of scalars, optional
        Standard deviation for Gabor kernel. The standard deviations of the
        Gabor filter are given for each axis as a sequence, or as a single
        number, in which case it is equal for all axes. These directions
        apply to the kernel *before* rotation.
    n_stds : scalar, optional
        The linear size of the kernel is n_stds (3 by default) standard
        deviations.
    offset : float, optional
        Phase offset of harmonic function in radians.
    axes : int or sequence of int
        Ordering of axes that defines the plane with regards to
        orientation. Ordering not specified will be padded with
        remaining axes in ascending order. Non-iterable values
        will be treated as the single element of a tuple.
        For classical cartesian ordering `(x, y, ...)`, set to `1`.
    ndim : int, optional
        Dimensionality of the kernel.

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
               'gabor_kernel(frequency, sigma=(sigma_x, sigma_y)).')

    if sigma_y is not None:
        warn(message)
        if 'sigma_x' in kwargs:
            sigma = (sigma_y, kwargs['sigma_x'])
        else:
            sigma = (sigma_y, sigma)

    # handle translation
    if not isinstance(theta, coll.Iterable):
        theta = (theta,) * (ndim - 1)

    if not isinstance(sigma, coll.Iterable):
        sigma = np.array([sigma] * ndim)
    else:
        sigma = np.append(sigma, [None] * (ndim - len(sigma)))
    default_sigma = _sigma_prefactor(bandwidth) / frequency
    sigma[sigma == None] = default_sigma  # noqa
    sigma = sigma.astype(None)

    coords = _decompose_quasipolar_coords(1, theta, axes)
    base_axis = np.zeros(ndim)
    base_axis[0] = 1
    rot = _compute_rotation_matrix(base_axis, coords)

    # calculate & rotate kernel size
    spatial_size = np.ceil(np.max(np.abs(n_stds * sigma * rot), axis=-1))
    spatial_size[spatial_size < 1] = 1

    # create mesh grid
    m = np.mgrid.__getitem__([slice(-c, c + 1) for c in spatial_size])

    rotm = np.matmul(m.T, rot).T

    gauss = _gaussian_kernel(rotm, sigma=sigma, center=0)

    compm = np.matmul(m.T, frequency * coords).T

    # complex harmonic function
    harmonic = np.exp(1j * (2 * np.pi * compm.sum(axis=0) + offset))

    g = np.zeros(m[0].shape, dtype=np.complex)
    g[:] = gauss * harmonic

    return g


def gabor(image, frequency=None, theta=0, bandwidth=1, sigma=None, sigma_y=None,
          n_stds=3, offset=0, mode='reflect', cval=0, kernel=None, **kwargs):
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
        deviations.
    offset : float, optional
        Phase offset of harmonic function in radians.
    mode : {'constant', 'nearest', 'reflect', 'mirror', 'wrap'}, optional
        Mode used to convolve image with a kernel, passed to `ndi.convolve`
    cval : scalar, optional
        Value to fill past edges of input if `mode` of convolution is
        'constant'. The parameter is passed to `ndi.convolve`.
    kernel : complex array
        Pre-computed gabor kernel. When applying the same filter to many
        images, using a kernel generated from `gabor_kernel` and passing it
        here may see significant computational improvements.

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
