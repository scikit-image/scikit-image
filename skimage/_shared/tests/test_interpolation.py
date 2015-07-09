from skimage._shared.interpolation import coord_map_py
from numpy.testing import assert_array_equal


def test_coord_map():
    reflect = [coord_map_py(4, n, 'R') for n in range(-6, 6)]
    expected_reflect = [2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2]
    assert_array_equal(reflect, expected_reflect)

    wrap = [coord_map_py(4, n, 'W') for n in range(-6, 6)]
    expected_wrap = [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    assert_array_equal(wrap, expected_wrap)

    nearest = [coord_map_py(4, n, 'N') for n in range(-6, 6)]
    expected_neareset = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3]
    assert_array_equal(nearest, expected_neareset)

    mirror = [coord_map_py(4, n, 'M') for n in range(-6, 6)]
    expected_mirror = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1]
    assert_array_equal(mirror, expected_mirror)

    other = [coord_map_py(4, n, 'undefined') for n in range(-6, 6)]
    assert_array_equal(other, list(range(-6, 6)))
