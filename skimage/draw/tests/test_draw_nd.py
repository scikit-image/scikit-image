from skimage.draw import line_nd
from skimage._shared.testing import assert_equal


def test_empty_line():
    coords = line_nd((1, 1, 1), (1, 1, 1))
    assert len(coords) == 3
    assert all(len(c) == 0 for c in coords)


def test_zero_line():
    coords = line_nd((-1, -1), (2, 2))
    assert_equal(coords, [[-1, 0, 1], [-1, 0, 1]])
