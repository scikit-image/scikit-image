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


def ellipsoid_coords(d, r, c, d_radius, r_radius, c_radius, shape=None,
                     rot_z=0., rot_y=0., rot_x=0., spacing=(1., 1., 1.)):
    """Generate coordinates of voxels within ellipsoid.

    Parameters
    ----------
    d, r, c : float
        Center coordinate of ellipsoid. The values will be divided by the
        spacing parameter inside the function.
    d_radius, r_radius, c_radius : float
        Radii of ellipsoid. The values will be divided by the spacing
        parameter inside the function.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for ellipsoids which exceed the
        image size.
        By default the full extent of the ellipsoid is used.
    rot_z, rot_y, rot_x : float, optional (default 0.)
        Rotation for each axis.
    spacing : (3,) array-like of float, optional (default (1., 1., 1.))
        Spacing in (z, y, x) spatial dimensions in this order.

    Returns
    -------
    dd, rr, cc : ndarray of int
        Voxel coordinates of ellipsoid.
        May be used to directl index into an array, e.g.
        ``img[dd, rr, cc] = 1``
    """

    center = np.array([d, r, c])
    radii = np.array([d_radius, r_radius, c_radius])

    # calculate a rotation matrix
    rot_z %= np.pi
    rot_y %= np.pi
    rot_x %= np.pi
    sin_rot_z, cos_rot_z = np.sin(rot_z), np.cos(rot_z)
    sin_rot_y, cos_rot_y = np.sin(rot_y), np.cos(rot_y)
    sin_rot_x, cos_rot_x = np.sin(rot_x), np.cos(rot_x)
    rotation_z = np.array([[1, 0, 0],
                           [0, cos_rot_z, sin_rot_z],
                           [0, -sin_rot_z, cos_rot_z]])
    rotation_y = np.array([[cos_rot_y, 0, -sin_rot_y],
                           [0, 1, 0],
                           [sin_rot_y, 0, cos_rot_y]])
    rotation_x = np.array([[cos_rot_x, sin_rot_x, 0],
                           [-sin_rot_x, cos_rot_x, 0],
                           [0, 0, 1]])
    rotation = rotation_z @ rotation_y @ rotation_x

    if len(spacing) != 3:
        raise ValueError(f'len(spacing) should be 3 but got {len(spacing)}')
    spacing = np.array(spacing)
    scaled_center = center / spacing

    # The upper_left_bottom and lower_right_top corners of the smallest cuboid
    # containing the ellipsoid.
    factor = np.array([
        [i, j, k] for k in (-1, 1) for j in (-1, 1) for i in (-1, 1)]).T
    radii_rot = np.abs(
        np.diag(1. / spacing) @ (rotation @ (np.diag(radii) @ factor))
    ).max(axis=1)
    upper_left_bottom = np.ceil(scaled_center - radii_rot).astype(int)
    lower_right_top = np.floor(scaled_center + radii_rot).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_ight by shape boundary.
        upper_left_bottom = np.maximum(upper_left_bottom, np.array([0, 0, 0]))
        lower_right_top = np.minimum(lower_right_top, np.array(shape[:3]) - 1)

    bounding_shape = lower_right_top - upper_left_bottom + 1

    d_lim, r_lim, c_lim = np.ogrid[0:float(bounding_shape[0]),
                                   0:float(bounding_shape[1]),
                                   0:float(bounding_shape[2])]
    d_org, r_org, c_org = scaled_center - upper_left_bottom
    d_rad, r_rad, c_rad = radii
    rotation_inv = np.linalg.inv(rotation)
    conversion_matrix = rotation_inv.dot(np.diag(spacing))
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
