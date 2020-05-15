import numpy as np


def _normalize(x):
    """Scale an array to have norm=1.

    Parameters
    ----------
    x : array-like
        Array to normalize.

    Returns
    -------
    u : ndarray
        Unitary array.

    Examples
    --------
    >>> x = np.arange(5)
    >>> uX = _normalize(x)
    >>> np.isclose(np.linalg.norm(uX), 1)
    True
    """
    v = np.asarray(x)

    norm = np.linalg.norm(v)

    return v / norm


def _axis_0_rotation_matrix(unit_vector, indices=None):
    """Generate a matrix that rotates a vector to be collinear with axis 0.

    Parameters
    ----------
    unit_vector : (N, ) array-like
        Unit vector.
    indices : sequence of int, optional
        Indices of the components of `unit_vector` that should be transformed.
        If `None`, defaults to all of the indices of `unit_vector`.

    Returns
    -------
    rotation_matrix : (N, N) ndarray
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
    >>> rotation_matrix = _axis_0_rotation_matrix([0, 1, 0])
    >>> rotation_matrix @ [0, 1, 0]
    array([ 1.,  0.,  0.])
    """
    unit_vector = np.array(unit_vector)  # copy it since it will be mutated
    ndim = len(unit_vector)

    if indices is None:
        indices = list(range(ndim))

    rotation_matrix = np.eye(ndim)

    # loop to create stages of 2D rotations around fixed axes
    # that are multiplied to form our nD matrix; see: [2]_
    for step in np.round(2 ** np.arange(np.log2(ndim))).astype(int):
        plane_rotation_matrix = np.eye(ndim)

        for n in range(0, ndim - step, step * 2):
            if n + step >= len(indices):
                break

            # axes that make up this plane
            i = indices[n]
            j = indices[n + step]

            # distance from origin in this plane
            radius = np.hypot(unit_vector[i], unit_vector[j])

            if radius > 0:
                # calculation of coefficients
                pcos = unit_vector[i] / radius
                psin = -unit_vector[j] / radius

                # base 2-dimensional rotation for this plane
                plane_rotation_matrix[i, i] = pcos
                plane_rotation_matrix[i, j] = -psin
                plane_rotation_matrix[j, i] = psin
                plane_rotation_matrix[j, j] = pcos

                unit_vector[i] = radius
                unit_vector[j] = 0

        # compound current plane's rotation with previous ones'
        rotation_matrix = plane_rotation_matrix @ rotation_matrix

    return rotation_matrix


def compute_rotation_matrix(src, dst, homogeneous_coords=False):
    """Generate a matrix to rotate one vector onto another.

    The MNMRG algorithm [1]_, implemented here, is summarized as:
        1. normalize directional vectors ``X`` and ``Y``
        2. initialize vector ``w`` containing the
           indices of the differences between ``X`` and ``Y``
        3. generate matrices ``Mx`` and ``My`` for the rotation
           of ``X`` and ``Y`` to the same axis for all indices
           in ``w``
        4. multiply the inverse of ``My`` by ``Mx`` to form
           the rotation matrix ``M`` which rotates vector ``X`` to the
           direction of vector ``Y``

    Parameters
    ----------
    src : (N, ) array-like
        Vector to rotate.
    dst : (N, ) array-like
        Vector of desired direction.
    homogeneous_coords : bool, optional
        Whether the input vectors should be treated
        as homogeneous coordinates [3]_.

    Returns
    -------
    rotation_matrix : (N, N) ndarray
        Matrix that rotates ``src`` onto ``dst``.

    References
    ----------
    .. [1] Ognyan Ivanov Zhelezov. One Modification which Increases Performance
           of N-Dimensional Rotation Matrix Generation Algorithm. International
           Journal of Chemistry, Mathematics, and Physics, Vol. 2 No. 2, 2018:
           pp. 13-18. https://dx.doi.org/10.22161/ijcmp.2.2.1
    .. [2] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    .. [3] https://en.wikipedia.org/wiki/Homogeneous_coordinates

    Examples
    --------
    >>> src = np.asarray([1, 0])
    >>> dst = np.asarray([.5, .5])
    >>> rotation_matrix = compute_rotation_matrix(src, dst)
    >>> src_rotated = rotation_matrix @ src
    >>> dst_normalized = dst / np.linalg.norm(dst)
    >>> np.allclose(src_rotated, dst_normalized)
    True
    """
    # step 1: vectors are normalized
    homogeneous_slice = -1 if homogeneous_coords else None
    src = _normalize(src[:homogeneous_slice])
    dst = _normalize(dst[:homogeneous_slice])

    if homogeneous_coords:
        src = np.append(src, 1)
        dst = np.append(dst, 1)

    # step 2: a vector is created containing the
    #         indices of difference between input vectors
    indices = np.flatnonzero(~np.isclose(src, dst))

    # step 3: matrices are generated for each input vector
    #         to rotate respective vector to the 0th axis
    src_rotation_matrix = _axis_0_rotation_matrix(src, indices)
    dst_rotation_matrix = _axis_0_rotation_matrix(dst, indices)

    # step 4: by rotating both vectors to the same direction
    #         and inverting one operation, a final
    #         rotation matrix is created
    # a rotation matrix is orthogonal, so its inverse is its transpose
    dst_rotation_matrix_inverse = dst_rotation_matrix.T

    rotation_matrix = dst_rotation_matrix_inverse @ src_rotation_matrix

    return rotation_matrix
