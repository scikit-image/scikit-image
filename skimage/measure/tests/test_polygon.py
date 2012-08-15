import numpy as np
from skimage.measure import approximate_polygon, subdivide_polygon


square = np.array([
    [0, 0], [0, 1], [0, 2], [0, 3],
    [1, 3], [2, 3], [3, 3],
    [3, 2], [3, 1], [3, 0],
    [2, 0], [1, 0], [0, 0]
])


def test_approximate_polygon():
    out = approximate_polygon(square, 0.1)
    np.testing.assert_array_equal(out, square[(0, 3, 6, 9, 12), :])

    out = approximate_polygon(square, 2.2)
    np.testing.assert_array_equal(out, square[(0, 6, 12), :])

    out = approximate_polygon(square[(0, 1, 3, 4, 5, 6, 7, 9, 11, 12), :], 0.1)
    np.testing.assert_array_equal(out, square[(0, 3, 6, 9, 12), :])

    out = approximate_polygon(square, -1)
    np.testing.assert_array_equal(out, square)


def test_subdivide_polygon():
    for degree in range(1, 7):
        out = subdivide_polygon(square, degree)
        np.testing.assert_array_equal(out[-1], out[0])
        np.testing.assert_equal(out.shape[0], 2 * square.shape[0] - 1)


if __name__ == "__main__":
    np.testing.run_module_suite()
