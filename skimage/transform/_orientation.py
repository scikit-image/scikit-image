import numpy as np


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
    """Generate a matrix that rotates a vector to coincide with the 0th (y-)
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


    Examples
    --------
    >>> R = _axis_0_rotation_matrix([0, 1, 0])
    >>> R @ [0, 1, 0]
    array([ 1.,  0.,  0.])
    """
    ndim = len(u)

    if indices is None:
        indices = range(ndim)

    x = u
    w = indices

    R = np.eye(ndim)

    # loop to create stages of 2D rotations around fixed axes
    # that are multiplied to form our nD matrix; see: [2]_
    for step in np.round(2 ** np.arange(np.log2(ndim))).astype(int):
        A = np.eye(ndim)

        for n in range(0, ndim - step, step * 2):
            if n + step >= len(w):
                break

            i = w[n]
            j = w[n + step]

            r = np.hypot(x[i], x[j])
            if r > 0:
                # calculation of coefficients
                pcos = x[i] / r
                psin = -x[j] / r

                # base 2-dimensional rotation
                A[i, i] = pcos
                A[i, j] = -psin
                A[j, i] = psin
                A[j, j] = pcos

                x[i] = r
                x[j] = 0

        R = A @ R  # multiply R by current matrix of stage A

    return R


def convert_quasipolar_coords(r, thetas):
    r"""Convert quasipolar coordinates to their Cartesian equivalents.

    Quasipolar coordinate conversion [1]_ is defined as follows:

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
    This conversion is denoted "quasipolar" because it is similar to polar
    coordinate conversion with a difference in the order that indices are
    interpreted.

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
    >>> convert_quasipolar_coords(1, [0])
    array([ 0.,  1.])

    >>> convert_quasipolar_coords(10, [np.pi / 2, 0])
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


def compute_rotation_matrix(src, dst, use_homogeneous_coords=False):
    """Generate a matrix for the rotation of one vector to the direction
    of another.

    The MNMRG algorithm [1]_ can be summarized as follows:
        1. directional vectors ``X`` and ``Y`` are normalized
        2. a vector ``w`` is initialized containing the
           indices of the differences between ``X`` and ``Y``
        3. matrices ``Mx`` and ``My`` are generated for the rotation
           of ``X`` and ``Y`` to the same axis for all indices
           in ``w``
        4. the inverse of ``My`` is mulitplied by ``Mx`` to form
           the rotation matrix ``M`` which rotates vector ``X`` to the
           direction of vector ``Y``

    Parameters
    ----------
    src : (N, ) array
        Vector to rotate.
    dst : (N, ) array
        Vector of desired direction.
    use_homogeneous_coords : bool, optional
        Whether the input vectors should be treated as homogeneous coordinates.

    Returns
    -------
    M : (N, N) array
        Matrix that rotates ``src`` to coincide with ``dst``.

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
    >>> Y = np.asarray([.5, .5])
    >>> M = compute_rotation_matrix(X, Y)
    >>> Z = M @ X
    >>> uY = Y / np.linalg.norm(Y)
    >>> np.allclose(Z, uY)
    True
    """
    # step 1: vectors are normalized
    homogeneous_slice = -use_homogeneous_coords or None
    X = _normalize(src[:homogeneous_slice])
    Y = _normalize(dst[:homogeneous_slice])

    if use_homogeneous_coords:
        X = np.append(X, 1)
        Y = np.append(Y, 1)

    # step 2: a vector is created containing the
    #         indices of difference between input vectors
    w = np.flatnonzero(~np.isclose(X, Y))

    # step 3: matrices are generated for each input vector
    #         to rotate respective vector to the 0th axis
    Mx = _axis_0_rotation_matrix(X, w)
    My = _axis_0_rotation_matrix(Y, w)

    # step 4: by rotating both vectors to the same direction
    #         and inverting one operation, a final
    #         rotation matrix is created
    My_inverse = My.T  # since My is orthogonal, its inverse is its transpose

    M = My_inverse @ Mx

    return M
