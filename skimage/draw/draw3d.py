import numpy as np
from scipy.special import elliprg


def ellipsoid(
    a, b, c, spacing=(1.0, 1.0, 1.0), rotation=(0.0, 0.0, 0.0), levelset=False
):
    """
    Generates ellipsoid with semimajor axes aligned with grid dimensions
    on grid with specified `spacing`.

    Parameters
    ----------
    a : float
        Length of semimajor axis aligned with plane-axis, coming out of plane.
    b : float
        Length of semimajor axis aligned with row-axis, going vertically down.
    c : float
        Length of semimajor axis aligned with column-axis, going horizontally right.
    spacing : 3-tuple of floats
        Spacing in three spatial dimensions.
    rotation : 3-tuple of floats, optional (default (0.0, 0.0, 0.0))
        Set the ellipsoid rotation in Euler angles (in radians) in the order (phi, theta, psi),
        where rotation angles are defined in counter-clockwise direction using the right-hand rule.
        `phi` about plane-axis, `theta` about intermediate row-axis, and `psi` about the final plane-axis.
    levelset : bool
        If True, returns the level set for this ellipsoid (signed level
        set about zero, with positive denoting interior) as np.float64.
        False returns a binarized version of said level set.

    Returns
    -------
    ellipsoid : (M, N, P) array
        Ellipsoid centered in a correctly sized array for given `spacing`.
        Boolean dtype unless `levelset=True`, in which case a float array is
        returned with the level set above 0.0 representing the ellipsoid.

    Notes
    -----
    The returned ellipsoid satisfies the following equation::
        ((plane_rot / a) ** 2 + (row_rot / float(b)) ** 2 + (col_rot / float(c)) ** 2) = 1

    where plane_rot, row_rot, and col_rot are the passive rotated coordinates generated using the Rotation matrix `R`::
        [plane_rot, row_rot, col_rot].T = R[2,0,1].T @ [plane, row, col].T

    where R is the standard active rotation matrix for the given Euler angles (theta, phi, psi) and coordinates (x, y, z) [1]_::
        R = [[cos(psi)*cos(phi) - cos(theta)*sin(phi)*sin(psi), -sin(psi)*cos(phi) - cos(theta)*sin(phi)*cos(psi), sin(theta)*sin(phi)],
             [cos(psi)*sin(phi) + cos(theta)*cos(phi)*sin(psi), -sin(psi)*sin(phi) + cos(theta)*cos(phi)*cos(psi), -sin(theta)*cos(phi)],
             [sin(psi)*sin(theta), cos(psi)*sin(theta), cos(theta)]]

    References
    ----------
    .. [1] Weisstein, Eric W. "Euler Angles." From MathWorld--A Wolfram Web Resource.
            https://mathworld.wolfram.com/EulerAngles.html

    """
    if (a <= 0) or (b <= 0) or (c <= 0):
        raise ValueError('Parameters a, b, and c must all be > 0')

    offset = np.r_[1, 1, 1] * np.r_[spacing]
    phi, theta, psi = rotation
    # Define the standard rotation matrix for the given Euler angles defined for the coordinate system (x,y,z)
    standard_rotation_matrix = np.array(
        [
            [
                np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.sin(psi),
                np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi),
                np.sin(psi) * np.sin(theta),
            ],
            [
                -np.sin(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(psi),
                -np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.cos(psi),
                np.cos(psi) * np.sin(theta),
            ],
            [np.sin(theta) * np.sin(phi), -np.sin(theta) * np.cos(phi), np.cos(theta)],
        ]
    )
    # Switch the elemets of the rotation matrix to match our convention of (plane, row, col)-axis <-> (z, x, y)-axis
    permutation = [2, 0, 1]
    rotation_matrix = standard_rotation_matrix[permutation, :][:, permutation]

    # Rotate the ellipsoid axes i.e. active rotation
    abc_rot = rotation_matrix @ np.r_[a, b, c]
    a_rot, b_rot, c_rot = np.abs(abc_rot)
    # Calculate limits, and ensure output volume is odd & symmetric
    low = np.ceil(-np.r_[a_rot, b_rot, c_rot] - offset)
    high = np.floor(np.r_[a_rot, b_rot, c_rot] + offset + 1)

    for dim in range(3):
        if (high[dim] - low[dim]) % 2 == 0:
            low[dim] -= 1
        num = np.arange(low[dim], high[dim], spacing[dim])
        if 0 not in num:
            low[dim] -= np.max(num[num < 0])

    # Generate (anisotropic) spatial grid
    plane_grid, row_grid, col_grid = np.mgrid[
        low[0] : high[0] : spacing[0],
        low[1] : high[1] : spacing[1],
        low[2] : high[2] : spacing[2],
    ]
    prc_grid = np.vstack((plane_grid.flatten(), row_grid.flatten(), col_grid.flatten()))
    # Rotate the grid i.e. passive rotation
    prc_grid_rot = rotation_matrix.T @ prc_grid

    # Extract the rotated coordinates
    plane_grid_rot = prc_grid_rot[0, :].reshape(plane_grid.shape)
    row_grid_rot = prc_grid_rot[1, :].reshape(row_grid.shape)
    col_grid_rot = prc_grid_rot[2, :].reshape(col_grid.shape)

    if not levelset:
        arr = (
            (plane_grid_rot / float(a)) ** 2
            + (row_grid_rot / float(b)) ** 2
            + (col_grid_rot / float(c)) ** 2
        ) <= 1
    else:
        arr = (
            (plane_grid_rot / float(a)) ** 2
            + (row_grid_rot / float(b)) ** 2
            + (col_grid_rot / float(c)) ** 2
        ) - 1

    return arr


def ellipsoid_stats(a, b, c):
    """Calculate analytical volume and surface area of an ellipsoid.

    The surface area of an ellipsoid is given by

    .. math:: S=4\\pi b c R_G\\!\\left(1, \\frac{a^2}{b^2}, \\frac{a^2}{c^2}\\right)

    where :math:`R_G` is Carlson's completely symmetric elliptic integral of
    the second kind [1]_. The latter is implemented as
    :py:func:`scipy.special.elliprg`.

    Parameters
    ----------
    a : float
        Length of semi-axis along x-axis.
    b : float
        Length of semi-axis along y-axis.
    c : float
        Length of semi-axis along z-axis.

    Returns
    -------
    vol : float
        Calculated volume of ellipsoid.
    surf : float
        Calculated surface area of ellipsoid.

    References
    ----------
    .. [1] Paul Masson (2020). Surface Area of an Ellipsoid.
           https://analyticphysics.com/Mathematical%20Methods/Surface%20Area%20of%20an%20Ellipsoid.htm

    """
    if (a <= 0) or (b <= 0) or (c <= 0):
        raise ValueError('Parameters a, b, and c must all be > 0')

    # Volume
    vol = 4 / 3.0 * np.pi * a * b * c

    # Surface area
    surf = 3 * vol * elliprg(1 / a**2, 1 / b**2, 1 / c**2)

    return vol, surf
