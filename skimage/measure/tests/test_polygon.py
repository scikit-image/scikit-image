import numpy as np
from skimage.measure import approximate_polygon, subdivide_polygon
from skimage.measure._polygon import _SUBDIVISION_MASKS

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
    out = approximate_polygon(square, 0)
    np.testing.assert_array_equal(out, square)


def test_subdivide_polygon():
    new_square1 = square
    new_square2 = square[:-1]
    new_square3 = square[:-1]
    # test iterative subdvision
    for _ in range(10):
        square1, square2, square3 = new_square1, new_square2, new_square3
        # test different B-Spline degrees
        for degree in range(1, 7):
            mask_len = len(_SUBDIVISION_MASKS[degree][0])
            # test circular
            new_square1 = subdivide_polygon(square1, degree)
            np.testing.assert_array_equal(new_square1[-1], new_square1[0])
            np.testing.assert_equal(new_square1.shape[0],
                                    2 * square1.shape[0] - 1)
            # test non-circular
            new_square2 = subdivide_polygon(square2, degree)
            np.testing.assert_equal(new_square2.shape[0],
                                    2 * (square2.shape[0] - mask_len + 1))
            # test non-circular, preserve_ends
            new_square3 = subdivide_polygon(square3, degree, True)
            np.testing.assert_equal(new_square3[0], square3[0])
            np.testing.assert_equal(new_square3[-1], square3[-1])

            np.testing.assert_equal(new_square3.shape[0],
                                    2 * (square3.shape[0] - mask_len + 2))

    # not supported B-Spline degree
    np.testing.assert_raises(ValueError, subdivide_polygon, square, 0)
    np.testing.assert_raises(ValueError, subdivide_polygon, square, 8)


if __name__ == "__main__":
    np.testing.run_module_suite()
