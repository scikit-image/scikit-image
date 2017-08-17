import numpy as np
from numpy.testing import assert_array_equal

from skimage.measure import points_in_poly, grid_points_in_poly


class test_npnpoly():
    def test_square(self):
        v = np.array([[0, 0],
                      [0, 1],
                      [1, 1],
                      [1, 0]])
        assert(points_in_poly([[0.5, 0.5]], v)[0])
        assert(not points_in_poly([[-0.1, 0.1]], v)[0])

    def test_triangle(self):
        v = np.array([[0, 0],
                      [1, 0],
                      [0.5, 0.75]])
        assert(points_in_poly([[0.5, 0.7]], v)[0])
        assert(not points_in_poly([[0.5, 0.76]], v)[0])
        assert(not points_in_poly([[0.7, 0.5]], v)[0])

    def test_type(self):
        assert(points_in_poly([[0, 0]], [[0, 0]]).dtype == np.bool)


def test_grid_points_in_poly():
    v = np.array([[0, 0],
                  [5, 0],
                  [5, 5]])

    expected = np.tril(np.ones((5, 5), dtype=bool))

    assert_array_equal(grid_points_in_poly((5, 5), v),
                       expected)

if __name__ == "__main__":
    np.testing.run_module_suite()
