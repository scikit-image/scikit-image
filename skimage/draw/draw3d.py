import numpy as np
from scipy.special import (ellipkinc as ellip_F, ellipeinc as ellip_E)


def ellipsoid(a, b, c, spacing=(1., 1., 1.), levelset=False):
    """
    Generates ellipsoid with semimajor axes aligned with grid dimensions
    on grid with specified `spacing`.

    Parameters
    ----------
    a : float
        Length of semimajor axis aligned with x-axis.
    b : float
        Length of semimajor axis aligned with y-axis.
    c : float
        Length of semimajor axis aligned with z-axis.
    spacing : tuple of floats, length 3
        Spacing in (x, y, z) spatial dimensions.
    levelset : bool
        If True, returns the level set for this ellipsoid (signed level
        set about zero, with positive denoting interior) as np.float64.
        False returns a binarized version of said level set.

    Returns
    -------
    ellip : (N, M, P) array
        Ellipsoid centered in a correctly sized array for given `spacing`.
        Boolean dtype unless `levelset=True`, in which case a float array is
        returned with the level set above 0.0 representing the ellipsoid.

    """
    if (a <= 0) or (b <= 0) or (c <= 0):
        raise ValueError('Parameters a, b, and c must all be > 0')

    offset = np.r_[1, 1, 1] * np.r_[spacing]

    # Calculate limits, and ensure output volume is odd & symmetric
    low = np.ceil(- np.r_[a, b, c] - offset)
    high = np.floor(np.r_[a, b, c] + offset + 1)

    for dim in range(3):
        if (high[dim] - low[dim]) % 2 == 0:
            low[dim] -= 1
        num = np.arange(low[dim], high[dim], spacing[dim])
        if 0 not in num:
            low[dim] -= np.max(num[num < 0])

    # Generate (anisotropic) spatial grid
    x, y, z = np.mgrid[low[0]:high[0]:spacing[0],
                       low[1]:high[1]:spacing[1],
                       low[2]:high[2]:spacing[2]]

    if not levelset:
        arr = ((x / float(a)) ** 2 +
               (y / float(b)) ** 2 +
               (z / float(c)) ** 2) <= 1
    else:
        arr = ((x / float(a)) ** 2 +
               (y / float(b)) ** 2 +
               (z / float(c)) ** 2) - 1

    return arr


def _rotate_axes_x(angle):
    r"""Rotate the axes of a coordinate system about the x-axis.

    .. math::

        R(\theta) = \begin{bmatrix}
                        \cos\theta & -\sin\theta & 0 \\
                        \sin\theta &  \cos\theta & 0 \\
                                 0 &           0 & 1 \\
                    \end{bmatrix}

    Parameters
    ----------
    angle : float
        Angle to rotate the axes about the x-axis.

    Returns
    -------
    rotation_matrix : (3, 3) ndarray of float
        Rotation matrix to rotate the axes about the x-axis by `angle`.
        The order of dimension is supposed to be ``(z, y, x)``.
    """

    sin, cos = np.sin(angle), np.cos(angle)
    return np.array([[cos, -sin, 0],
                     [sin, cos, 0],
                     [0, 0, 1]])


def _rotate_axes_y(angle):
    r"""Rotate the axes of a coordinate system about the y-axis.

    .. math::

        R(\theta) = \begin{bmatrix}
                         \cos\theta & 0 & \sin\theta \\
                                  0 & 1 &          0 \\
                        -\sin\theta & 0 & \cos\theta \\
                    \end{bmatrix}

    Parameters
    ----------
    angle : float
        Angle to rotate the axes about the y-axis.

    Returns
    -------
    rotation_matrix : (3, 3) ndarray of float
        Rotation matrix to rotate the axes about the y-axis by ``angle``.
        The order of dimension is supposed to be ``(z, y, x)``.
    """

    sin, cos = np.sin(angle), np.cos(angle)
    return np.array([[cos, 0, sin],
                     [0, 1, 0],
                     [-sin, 0, cos]])


def _rotate_axes_z(angle):
    r"""Rotate the axes of a coordinate system about the z-axis.

    .. math::

        R(\theta) = \begin{bmatrix}
                        1 &          0 &           0 \\
                        0 & \cos\theta & -\sin\theta \\
                        0 & \sin\theta &  \cos\theta \\
                    \end{bmatrix}

    Parameters
    ----------
    angle : float
        Angle to rotate the axes about the z-axis.

    Returns
    -------
    rotation_matrix : (3, 3) ndarray of float
        Rotation matrix to rotate the axes about the z-axis by ``angle``.
        The order of dimension is supposed to be ``(z, y, x)``.
    """

    sin, cos = np.sin(angle), np.cos(angle)
    return np.array([[1, 0, 0],
                     [0, cos, -sin],
                     [0, sin, cos]])


def _rotate_axes(about, angle):
    """Rotate the axes of a coordinate system.

    Parameters
    ----------
    about : {'x', 'y', 'z'},
        Axis to rotate about.

    angle : float
        Angle to rotate the axes.

    Returns
    -------
    rotation_matrix : (3, 3) ndarray of float
        Rotation matrix to rotate the axes about the z-axis by ``angle``.
        The order of dimension is supposed to be ``(z, y, x)``.
    """

    if about == 'x':
        return _rotate_axes_x(angle)
    elif about == 'y':
        return _rotate_axes_y(angle)
    elif about == 'z':
        return _rotate_axes_z(angle)
    else:
        raise ValueError(f'about={about} is not supported.')


def _angles_to_rotmat(angles, order='zxz', is_intrinsic=True):
    r"""Calculate a rotation matrix from the Euler angles.

    The Euler angles are defined based on ``order`` and ``is_intrinsic``.
    The following example shows the ``zxz`` intrinsic rotations.

    .. math::

        R(\phi) = \begin{bmatrix}
                      1 &        0 &         0 \\
                      0 & \cos\phi & -\sin\phi \\
                      0 & \sin\phi &  \cos\phi \\
                  \end{bmatrix}

        R(\theta) = \begin{bmatrix}
                        \cos\theta & -\sin\theta & 0 \\
                        \sin\theta &  \cos\theta & 0 \\
                                 0 &           0 & 1 \\
                    \end{bmatrix}

        R(\psi) = \begin{bmatrix}
                      1 &         0 &         0 \\
                      0 &  \cos\psi & -\sin\psi \\
                      0 &  \sin\psi &  \cos\psi \\
                  \end{bmatrix}

        \\

        R(\phi, \theta, \psi) = (R(\psi)R(\theta)R(\phi))^\intercal

    Parameters
    ----------
    angles : (3,) ndarray of float
        Euler angles ``(phi, theta, psi)`` in this order.

    order : str, length 3, optional (default 'zxz')
        Order of rotations. Each character should be one of ``{'x', 'y', 'z'}``
        and there should be no adjacent repeating characters.

    is_intrinsic : bool optional (default True)
        Specify if the rotations are intrinsic (``True``) or extrinsic
        (``False``).

        - *intrinsic*: rotations occur about the rotating coordinate system
        - *extrinsic*: rotations occur about the original coordinate system

    Returns
    -------
    rotation_matrix : (3, 3) ndarray of float
        Rotation matrix calculated from the Euler angles.
    """

    if (not isinstance(order, str) or len(order) != 3 or order.strip('xyz')
            or order[0] == order[1] or order[1] == order[2]):
        raise ValueError(f'order: {order} is invalid')
    angles %= np.pi
    phi, theta, psi = angles
    rotmat_phi = _rotate_axes(order[0], phi)
    rotmat_theta = _rotate_axes(order[1], theta)
    rotmat_psi = _rotate_axes(order[2], psi)
    if is_intrinsic:
        rotmat = (rotmat_psi @ rotmat_theta @ rotmat_phi).T
    else:
        rotmat = rotmat_psi.T @ rotmat_theta.T @ rotmat_phi.T
    return rotmat


def ellipsoid_coords(center, axis_lengths, *, shape=None, rotation_angles=None,
                     rotation_order='zxz', is_intrinsic=True, spacing=None):
    r"""Generate coordinates of voxels within ellipsoid.

    Parameters
    ----------
    center : (3,) array-like of float
        Center coordinate of ellipsoid. The order is (z, y, x). The values will
        be divided by ``spacing`` inside the function.
    axis_lengths : (3,) array-like of float
        Axis lengths of ellipsoid. The order is (z, y, x). The values will be
        divided by ``spacing`` inside the function.
    shape : tuple of int, length 3, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for ellipsoids which exceed the
        image size. By default the full extent of the ellipsoid is used.
    rotation_angles : (3,) array-like of float, optional (default None)
        Rotation angles to rotate the ellipsoid. The order of rotations is
        specified by ``rotation_order``. No rotation will be applied if
        ``rotation_angles`` is ``None``.
    rotation_order : str, length 3, optional (default 'zxz')
        Order of rotations. Each character should be one of ``{'x', 'y', 'z'}``
        and there should be no adjacent repeating characters.
    is_intrinsic : bool optional (default True)
        Specify if the rotations are intrinsic (``True``) or extrinsic
        (``False``).

        - *intrinsic*: rotations occur about the rotating coordinate system
        - *extrinsic*: rotations occur about the original coordinate system
    spacing : (3,) array-like of float, optional (default None)
        Spacing in each spatial dimension. The order is (z, y, x). If
        ``spacing`` is ``None``, ``1.`` is set to all dimensions.

    Returns
    -------
    dd, rr, cc : ndarray of int
        Voxel coordinates of ellipsoid.
        May be used to directl index into an array, e.g.
        ``img[dd, rr, cc] = 1``

    Raises
    ------
    ValueError
        If the length of ``center`` is not 3.
    ValueError
        If the length of ``axis_length`` is not 3.
    ValueError
        If the length of ``rotation_angles`` is not 3.
    ValueError
        If the length of ``spacing`` is not 3.
    ValueError
        If ``rotation_order`` is invalid.

    Notes
    ------
    ``rotation_angles`` are defined as the Euler angles based on
    ``rotation_order`` and ``is_intrinsic``. The following example shows the
    ``zxz`` intrinsic rotations.

    .. math::

        R(\phi) = \begin{bmatrix}
                      1 &        0 &         0 \\
                      0 & \cos\phi & -\sin\phi \\
                      0 & \sin\phi &  \cos\phi \\
                  \end{bmatrix}

        R(\theta) = \begin{bmatrix}
                        \cos\theta & -\sin\theta & 0 \\
                        \sin\theta &  \cos\theta & 0 \\
                                 0 &           0 & 1 \\
                    \end{bmatrix}

        R(\psi) = \begin{bmatrix}
                      1 &        0 &         0 \\
                      0 & \cos\psi & -\sin\psi \\
                      0 & \sin\psi &  \cos\psi \\
                           \end{bmatrix}

        \\

        R(\phi, \theta, \psi) = (R(\psi)R(\theta)R(\phi))^\intercal

    """

    if len(center) != 3:
        raise ValueError(f'len(center) should be 3 but got {len(center)}')
    center = np.array(center)

    if len(axis_lengths) != 3:
        raise ValueError(
            f'len(axis_lengths) should be 3 but got {len(axis_lengths)}')
    axis_lengths = np.array(axis_lengths)

    if rotation_angles is not None:
        if len(rotation_angles) != 3:
            raise ValueError(
                'len(rotation_angles) should be 3 but got',
                len(rotation_angles))
        if (not isinstance(rotation_order, str)
            or len(rotation_order) != 3
            or rotation_order.strip('xyz')
            or rotation_order[0] == rotation_order[1]
                or rotation_order[1] == rotation_order[2]):
            raise ValueError(f'order: {rotation_order} is invalid')
        rotation_angles = np.array(rotation_angles)
        rotmat = _angles_to_rotmat(rotation_angles, order=rotation_order,
                                   is_intrinsic=is_intrinsic)
    else:
        rotmat = np.eye(3)

    if spacing is None:
        spacing = np.ones(3)
    elif len(spacing) != 3:
        raise ValueError(f'len(spacing) should be 3 but got {len(spacing)}')
    spacing = np.array(spacing)
    scaled_center = center / spacing

    # The upper_left_bottom and lower_right_top corners of the smallest cuboid
    # containing the ellipsoid.
    factor = np.array([
        [i, j, k] for k in (-1, 1) for j in (-1, 1) for i in (-1, 1)]).T
    axis_lengths_rot = np.abs(
        np.diag(1. / spacing) @ (rotmat @ (np.diag(axis_lengths) @ factor))
    ).max(axis=1)
    upper_left_bottom = np.ceil(scaled_center - axis_lengths_rot).astype(int)
    lower_right_top = np.floor(scaled_center + axis_lengths_rot).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_ight by shape boundary.
        upper_left_bottom = np.maximum(upper_left_bottom, np.array([0, 0, 0]))
        lower_right_top = np.minimum(lower_right_top, np.array(shape[:3]) - 1)

    bounding_shape = lower_right_top - upper_left_bottom + 1

    d_lim, r_lim, c_lim = np.ogrid[0:float(bounding_shape[0]),
                                   0:float(bounding_shape[1]),
                                   0:float(bounding_shape[2])]
    d_org, r_org, c_org = scaled_center - upper_left_bottom
    d_rad, r_rad, c_rad = axis_lengths
    rotmat_inv = np.linalg.inv(rotmat)
    conversion_matrix = rotmat_inv @ np.diag(spacing)
    d, r, c = (d_lim - d_org), (r_lim - r_org), (c_lim - c_org)
    distances = (
        ((d * conversion_matrix[0, 0]
          + r * conversion_matrix[0, 1]
          + c * conversion_matrix[0, 2]) / d_rad) ** 2 +
        ((d * conversion_matrix[1, 0]
          + r * conversion_matrix[1, 1]
          + c * conversion_matrix[1, 2]) / r_rad) ** 2 +
        ((d * conversion_matrix[2, 0]
          + r * conversion_matrix[2, 1]
          + c * conversion_matrix[2, 2]) / c_rad) ** 2
    )
    if distances.size == 0:
        return (np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int))
    dd, rr, cc = np.nonzero(distances < 1)
    dd.flags.writeable = True
    rr.flags.writeable = True
    cc.flags.writeable = True
    dd += upper_left_bottom[0]
    rr += upper_left_bottom[1]
    cc += upper_left_bottom[2]
    return dd, rr, cc


def ellipsoid_stats(a, b, c):
    """
    Calculates analytical surface area and volume for ellipsoid with
    semimajor axes aligned with grid dimensions of specified `spacing`.

    Parameters
    ----------
    a : float
        Length of semimajor axis aligned with x-axis.
    b : float
        Length of semimajor axis aligned with y-axis.
    c : float
        Length of semimajor axis aligned with z-axis.

    Returns
    -------
    vol : float
        Calculated volume of ellipsoid.
    surf : float
        Calculated surface area of ellipsoid.

    """
    if (a <= 0) or (b <= 0) or (c <= 0):
        raise ValueError('Parameters a, b, and c must all be > 0')

    # Calculate volume & surface area
    # Surface calculation requires a >= b >= c and a != c.
    abc = [a, b, c]
    abc.sort(reverse=True)
    a = abc[0]
    b = abc[1]
    c = abc[2]

    # Volume
    vol = 4 / 3. * np.pi * a * b * c

    # Analytical ellipsoid surface area
    phi = np.arcsin((1. - (c ** 2 / (a ** 2.))) ** 0.5)
    d = float((a ** 2 - c ** 2) ** 0.5)
    m = (a ** 2 * (b ** 2 - c ** 2) /
         float(b ** 2 * (a ** 2 - c ** 2)))
    F = ellip_F(phi, m)
    E = ellip_E(phi, m)

    surf = 2 * np.pi * (c ** 2 +
                        b * c ** 2 / d * F +
                        b * d * E)

    return vol, surf
