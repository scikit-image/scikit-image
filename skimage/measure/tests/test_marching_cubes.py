import numpy as np
from numpy.testing import assert_raises
from scipy.special import (ellipkinc as ellip_F, ellipeinc as ellip_E)

from skimage.measure import marching_cubes, mesh_surface_area


def _ellipsoid(a, b, c, sampling=(1., 1., 1.), info=False, tight=False,
               levelset=False):
    """
    Generates ellipsoid with semimajor axes aligned with grid dimensions,
    on grid with specified `sampling`.

    Parameters
    ----------
    a : float
        Length of semimajor axis aligned with x-axis
    b : float
        Length of semimajor axis aligned with y-axis
    c : float
        Length of semimajor axis aligned with z-axis
    sampling : tuple of floats, length 3
        Sampling in each spatial dimension
    info : bool
        If False, only `bool_arr` returned.
        If True, (`bool_arr`, `vol`, `surf`) returned; the additional
        values are analytical volume and surface area calculated for
        this ellipsoid.
    tight : bool
        Controls if the ellipsoid will precisely be contained within
        the returned volume (tight=True) or if each dimension will be
        2 longer than necessary (tight=False). For algorithms which
        need both sides of a contour, use False.
    levelset : bool
        If True, returns the level set for this ellipsoid (signed level
        set about zero, with positive denoting interior) as np.float64.
        False returns a binarized version of said level set.

    Returns
    -------
    bool_arr : (N, M, P) array
        Sphere in an appropriately sized boolean array.
    vol : float
        Analytically calculated volume of ellipsoid. Only returned if
        `info` is True.
    surf : float
        Analytically calculated surface area of ellipsoid. Only returned
        if `info` is True.

    """
    if not tight:
        offset = np.r_[1, 1, 1] * np.r_[sampling]
    else:
        offset = np.r_[0, 0, 0]

    # Calculate limits, and ensure output volume is odd & symmetric
    low = np.ceil((-np.r_[a, b, c] - offset))
    high = np.floor((np.r_[a, b, c] + offset + 1))
    for dim in range(3):
        if (high[dim] - low[dim]) % 2 == 0:
            low[dim] -= 1
        num = np.arange(low[dim], high[dim], sampling[dim])
        if 0 not in num:
            low[dim] -= np.max(num[num < 0])

    # Generate (anisotropic) spatial grid
    x, y, z = np.mgrid[low[0]:high[0]:sampling[0],
                       low[1]:high[1]:sampling[1],
                       low[2]:high[2]:sampling[2]]

    if not levelset:
        arr = ((x / float(a)) ** 2 +
               (y / float(b)) ** 2 +
               (z / float(c)) ** 2) <= 1
    else:
        arr = ((x / float(a)) ** 2 +
               (y / float(b)) ** 2 +
               (z / float(c)) ** 2) - 1

    if not info:
        return arr
    else:
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

        return arr, vol, surf


def test_marching_cubes_isotropic():
    ellipsoid_isotropic, _, surf = _ellipsoid(6, 10, 16,
                                              levelset=True,
                                              info=True)
    verts, faces = marching_cubes(ellipsoid_isotropic, 0.)
    surf_calc = mesh_surface_area(verts, faces)

    # Test within 1% tolerance for isotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.99


def test_marching_cubes_anisotropic():
    sampling = (1., 10 / 6., 16 / 6.)
    ellipsoid_isotropic, _, surf = _ellipsoid(6, 10, 16,
                                              sampling=sampling,
                                              levelset=True,
                                              info=True)
    verts, faces = marching_cubes(ellipsoid_isotropic, 0.,
                                  sampling=sampling)
    surf_calc = mesh_surface_area(verts, faces)
    # Test within 1.5% tolerance for anisotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.985


def test_invalid_input():
    assert_raises(ValueError, marching_cubes, np.zeros((2, 2, 1)), 0)
    assert_raises(ValueError, marching_cubes, np.zeros((2, 2, 1)), 1)
    assert_raises(ValueError, marching_cubes, np.ones((3, 3, 3)), 1,
                  sampling=(1, 2))
    assert_raises(ValueError, marching_cubes, np.zeros((20, 20)), 0)


if __name__ == '__main__':
    np.testing.run_module_suite()
