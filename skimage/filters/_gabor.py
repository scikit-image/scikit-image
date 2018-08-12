import collections as coll
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import assert_nD
from warnings import warn


__all__ = ['gabor_kernel', 'gabor']


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
    >>> np.isclose(np.linalg.norm(uX), 1)
    True
    """
    u = np.asarray(x)

    norm = np.linalg.norm(u)

    return u / norm


def _axis_0_rotation_matrix(u, indices=None):
    """Generates a matrix that rotates a vector to coincide with the 0th (y-)
       coordinate axis.

    Parameters
    ----------
    u : (N, ) array
        Unit vector.
    indices : sequence of int, optional
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
    .. [2] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    """
    ndim = len(u)

    if indices is None:
        indices = range(ndim)

    x = u
    w = indices

    R = np.eye(ndim)  # Initial rotation matrix = Identity matrix

    # Loop to create matrices of stages
    # These stages are 2D rotations around fixed axes that are combined
    # together to form our nD matrix; see: [2]
    for step in np.round(2 ** np.arange(np.log2(ndim))).astype(int):
        A = np.eye(ndim)

        for n in range(0, ndim - step, step * 2):
            if n + step >= len(w):
                break

            i = w[n]
            j = w[n + step]

            r = np.hypot(x[i], x[j])
            if r > 0:
                # Calculation of coefficients
                pcos = x[i] / r
                psin = -x[j] / r

                # Base 2-dimensional rotation
                A[i, i] = pcos
                A[i, j] = -psin
                A[j, i] = psin
                A[j, j] = pcos

                x[i] = r
                x[j] = 0

        R = np.matmul(A, R)  # Multiply R by current matrix of stage A

    return R


def _compute_rotation_matrix(src, dst, use_homogeneous_coords=False):
    """Generates a matrix for the rotation of one vector to the direction
    of another.

    The MNMRG algorithm cam be described as follows:
    1. directional vectors X and Y are normalized
    2. a vector w is initialized containing the indices of the differences
       between X and Y
    3. matrices Mx and My for the rotation of X and Y to the same axis are
       generated
    4. the inverse of My is combined with Mx to form the rotation matrix M
       that rotates vector X to the direction of vector Y

    Parameters
    ----------
    src : (N, ) array
        Vector to rotate.
    dst : (N, ) array
        Vector of desired direction.
    use_homogeneous_coords : bool, optional
        If the input vectors should be treated as homogeneous coordinates.

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
    .. [2] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions

    Examples
    --------
    >>> X = np.asarray([1, 0])
    >>> Y = np.asarray([0.5, 0.5])

    >>> M = _compute_rotation_matrix(X, Y)
    >>> Z = M @ X

    >>> uY = Y / np.linalg.norm(Y)
    >>> np.allclose(Z, uY)
    True
    """
    homogeneous_slice = -use_homogeneous_coords or None
    X = _normalize(src[:homogeneous_slice])
    Y = _normalize(dst[:homogeneous_slice])

    if use_homogeneous_coords:
        X = np.append(X, 1)
        Y = np.append(Y, 1)

    w = np.flatnonzero(~np.isclose(X, Y))  # indices of difference

    Mx = _axis_0_rotation_matrix(X, w)
    My = _axis_0_rotation_matrix(Y, w)

    My_inverse = My.T  # since My is orthogonal its inverse is its transpose
    M = np.matmul(My_inverse, Mx)

    return M


def _convert_quasipolar_coords(r, thetas):
    """Converts quasipolar coordinates to their Cartesian equivalents.

    Quasipolar coordinate conversion is defined as follows:

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

    Parameters
    ----------
    r : float
        Radial coordinate.
    thetas : (N, ) array
        Quasipolar angles.

    Returns
    -------
    coords : (``N + 1``, ) array
        Cartesian conversion of the quasipolar coordinates.

    References
    ----------
    .. [1] Tan Mai Nguyen. N-Dimensional Quasipolar Coordinates - Theory and
           Application. University of Nevada: Las Vegas, Nevada, 2014.
           https://digitalscholarship.unlv.edu/thesesdissertations/2125

    Notes
    -----
    In terms of polar coordinate conversion:

    .. math::

         \left\{
         \begin{array}{ll}
                 y = x_0 = r \sin \theta_0 \\
                 x = x_1 = r \cos \theta_0
         \end{array}
         \right.

    In terms of spherical coordinate conversion:

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
    >>> _convert_quasipolar_coords(1, [0])
    array([ 0.,  1.])
    >>> _convert_quasipolar_coords(10, [np.pi / 2, 0])
    array([  0.,   0.,  10.])
    """
    num_axes = len(thetas) + 1
    coords = r * np.ones(num_axes)

    for which_theta, theta in enumerate(thetas[::-1]):
        sine = np.sin(theta)
        theta_index = num_axes - which_theta - 1

        for axis in range(theta_index):
            coords[axis] *= sine

        coords[theta_index] *= np.cos(theta)

    return coords


def _gaussian_kernel(coords, center=0, sigma=1):
    """Multi-dimensional Gaussian kernel.

    Parameters
    ----------
    coords : non-complex array
        Coordinate matrices.
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
    gauss : (``ndim``, ``ndim``)
        Filter kernel.

    References
    ----------
    .. [1] Bart M. ter Haar Romeny. The Gaussian Kernel. In: Front-End Vision
           and Multi-Scale Image Analysis. Computational Imaging and Vision,
           Vol. 27, Springer: Dordrecht, 2003: pp. 37-51.
           ISBN: 978-1-4020-8840-7
           https://doi.org/10.1007/978-1-4020-8840-7_3
    """
    ndim = np.shape(coords)[0]

    # normalization factor
    norm = (2 * np.pi) ** (ndim / 2) * np.prod(sigma)

    # center image
    coords = np.asarray(coords) - center

    scaled_coords = coords / np.reshape(sigma, [ndim] + [1] * ndim)

    # gaussian function
    gauss = np.exp(-0.5 * np.sum(scaled_coords ** 2, axis=0))

    return gauss / norm


def _sigma_prefactor(bandwidth):
    b = bandwidth
    # See http://www.cs.rug.nl/~imaging/simplecell.html
    return 1.0 / np.pi * np.sqrt(np.log(2) / 2.0) * \
        (2.0 ** b + 1) / (2.0 ** b - 1)


def gabor_kernel(frequency, theta=0, bandwidth=1, sigma=None, sigma_y=None,
                 n_stds=3, offset=0, axes=None, ndim=2, **kwargs):
    """Multi-dimensional complex Gabor kernel.

    A Gabor kernel is a Gaussian kernel modulated by a complex harmonic
    function. Harmonic function consists of an imaginary sine function
    and a real cosine function. Spatial frequency is inversely proportional
    to the wavelength of the harmonic and to the standard deviation of a
    Gaussian kernel. The bandwidth is also inversely proportional to the
    standard deviation.

    Parameters
    ----------
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
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
    axes : int or sequence of int, optional
        Ordering of axes that defines the plane of rotation. Ordering not
        specified will be padded with remaining axes in ascending order.
        Non-iterable values will be treated as the single element of a
        tuple. For classical cartesian ordering `(x, y, ...)`, set to `1`.
        Defaults to `1` but will default to `0` in scikit-image 0.17.
    ndim : int, optional
        Dimensionality of the kernel.

    Returns
    -------
    g : complex array
        Complex filter kernel.

    References
    ----------
    .. [1] Tie Yun and Ling Guan. Human Emotion Recognition Using Real 3D
           Visual Features from Gabor Library. 2010 IEEE International
           Workshop on Multimedia Signal Processing: Saint Malo, France, 2010.
           https://doi.org/10.1109/MMSP.2010.5662073
    .. [2] Yingjie Wang and Chin-Seng Chua. Face recognition from 2D and 3D
           images using 3D Gabor filters. School of Electrical and Electronic
           Engineering, Nanyang Technological University: Singapore,
           Singapore, 2005. https://doi.org/10.1016/j.imavis.2005.07.005

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
    if sigma_y is not None:
        warn('Using deprecated, 2D-only interface to gabor_kernel. '
             'This interface will be removed in scikit-image 0.17. Use '
             'gabor_kernel(frequency, sigma=(sigma_y, sigma_x)).')

        if 'sigma_x' in kwargs:
            sigma = (sigma_y, kwargs['sigma_x'])
        else:
            sigma = (sigma_y, sigma)

    if axes is None:
        warn('Default value of `axes` will be changed from `1` to `0` in '
             'scikit-image 0.17. Specify this argument to silence this '
             ' warning.')
        axes = 1

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

    if not isinstance(axes, coll.Iterable):
        axes = (axes,)
    axes = np.append(axes, np.setdiff1d(range(ndim), axes))

    coords = _convert_quasipolar_coords(1, theta)
    base_axis = (1,) + (0,) * (ndim - 1)
    rot = _compute_rotation_matrix(base_axis, coords[axes])

    # calculate rotated kernel size
    spatial_size = np.max(np.abs(n_stds * sigma * rot), axis=-1)
    spatial_size = np.ceil(spatial_size).astype(np.int)
    spatial_size[spatial_size < 1] = 1

    # create kernel space
    m = np.asarray(np.meshgrid(*[range(-c, c + 1) for c in spatial_size],
                               indexing='ij'))

    rotm = (m.T @ rot.T).T

    gauss = _gaussian_kernel(rotm, sigma=sigma, center=0)

    compm = frequency * m * np.reshape(coords, [ndim] + [1] * ndim)

    # complex harmonic function
    harmonic = np.exp(1j * (2 * np.pi * compm.sum(axis=0) + offset))

    g = np.zeros(m.shape[1:], dtype=np.complex)
    g[:] = gauss * harmonic

    return g


def gabor(image, frequency=None, theta=0, bandwidth=1, sigma=None,
          sigma_y=None, n_stds=3, offset=0, mode='reflect', cval=0,
          axes=1, kernel=None, **kwargs):
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
    image : non-complex array
        Input image.
    frequency : float, optional
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
    mode : {'constant', 'nearest', 'reflect', 'mirror', 'wrap'}, optional
        Mode used to convolve image with a kernel, passed to `ndi.convolve`
    cval : scalar, optional
        Value to fill past edges of input if `mode` of convolution is
        'constant'. The parameter is passed to `ndi.convolve`.
    axes : int or sequence of int, optional
        Ordering of axes that defines the plane with regards to
        orientation. Ordering not specified will be padded with
        remaining axes in ascending order. Non-iterable values
        will be treated as the single element of a tuple.
        For classical cartesian ordering `(x, y, ...)`, set to `1`.
    kernel : complex array, optional
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
    .. [1] Tie Yun and Ling Guan. Human Emotion Recognition Using Real 3D
           Visual Features from Gabor Library. 2010 IEEE International
           Workshop on Multimedia Signal Processing: Saint Malo, France, 2010.
           https://doi.org/10.1109/MMSP.2010.5662073
    .. [2] Yingjie Wang and Chin-Seng Chua. Face recognition from 2D and 3D
           images using 3D Gabor filters. School of Electrical and Electronic
           Engineering, Nanyang Technological University: Singapore,
           Singapore, 2005. https://doi.org/10.1016/j.imavis.2005.07.005

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
                              n_stds, offset, ndim=np.ndim(image), **kwargs)
    else:
        if frequency is not None:
            warn("gabor() received arguments of "
                 "both 'kernel' and 'frequency'; "
                 "'frequency' will be ignored")
        assert_nD(np.ndim(image), np.ndim(image))

    filtered_real = ndi.convolve(image, np.real(kernel), mode=mode, cval=cval)
    filtered_imag = ndi.convolve(image, np.imag(kernel), mode=mode, cval=cval)

    return filtered_real, filtered_imag
