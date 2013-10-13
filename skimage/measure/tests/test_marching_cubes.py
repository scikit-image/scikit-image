import numpy as np
from numpy.testing import assert_raises

from skimage.draw import ellipsoid, ellipsoid_stats
from skimage.measure import marching_cubes, mesh_surface_area


def test_marching_cubes_isotropic():
    ellipsoid_isotropic = ellipsoid(6, 10, 16, levelset=True)
    _, surf = ellipsoid_stats(6, 10, 16)
    verts, faces = marching_cubes(ellipsoid_isotropic, 0.)
    surf_calc = mesh_surface_area(verts, faces)

    # Test within 1% tolerance for isotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.99


def test_marching_cubes_anisotropic():
    spacing = (1., 10 / 6., 16 / 6.)
    ellipsoid_anisotropic = ellipsoid(6, 10, 16, spacing=spacing,
                                      levelset=True)
    _, surf = ellipsoid_stats(6, 10, 16)
    verts, faces = marching_cubes(ellipsoid_anisotropic, 0.,
                                  spacing=spacing)
    surf_calc = mesh_surface_area(verts, faces)

    # Test within 1.5% tolerance for anisotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.985


def test_invalid_input():
    assert_raises(ValueError, marching_cubes, np.zeros((2, 2, 1)), 0)
    assert_raises(ValueError, marching_cubes, np.zeros((2, 2, 1)), 1)
    assert_raises(ValueError, marching_cubes, np.ones((3, 3, 3)), 1,
                  spacing=(1, 2))
    assert_raises(ValueError, marching_cubes, np.zeros((20, 20)), 0)


if __name__ == '__main__':
    np.testing.run_module_suite()
