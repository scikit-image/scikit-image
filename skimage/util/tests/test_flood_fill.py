import numpy as np
from pytest import raises

from skimage.util import flood_fill

eps = 1e-12


def test_empty_input():
    output = flood_fill(np.empty(0), (), 2)
    assert output.dtype == np.uint8

    output = flood_fill(np.empty(0), (), 2, indices=True)
    assert output.dtype == np.intp


def test_float16():
    image = np.array([9., 0.1, 42], dtype=np.float16)
    with raises(TypeError, match="dtype of `image` is float16"):
            flood_fill(image, 0, 1)


def test_overrange_tolerance_int():
    image = np.arange(256, dtype=np.uint8).reshape((8, 8, 4))
    expected = np.zeros_like(image)

    output = flood_fill(image, (7, 7, 3), 0, tolerance=379)

    np.testing.assert_equal(output, expected)


def test_overrange_tolerance_float():
    max_value = np.finfo(np.float32).max
    min_value = np.finfo(np.float32).min

    image = np.random.uniform(size=(64, 64), low=-1., high=1.).astype(
        np.float32)
    image *= max_value

    expected = np.ones_like(image)
    output = flood_fill(image, (0, 1), 1., tolerance=max_value*10)

    np.testing.assert_equal(output, expected)


def test_inplace_int():
    image = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 2, 2, 0],
                      [0, 1, 1, 0, 2, 2, 0],
                      [1, 0, 0, 0, 0, 0, 3],
                      [0, 1, 1, 1, 3, 3, 4]])

    flood_fill(image, (0, 0), 5, inplace=True)

    expected = np.array([[5, 5, 5, 5, 5, 5, 5],
                        [5, 1, 1, 5, 2, 2, 5],
                        [5, 1, 1, 5, 2, 2, 5],
                        [1, 5, 5, 5, 5, 5, 3],
                        [5, 1, 1, 1, 3, 3, 4]])

    np.testing.assert_array_equal(image, expected)


def test_inplace_float():
    image = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 2, 2, 0],
                      [0, 1, 1, 0, 2, 2, 0],
                      [1, 0, 0, 0, 0, 0, 3],
                      [0, 1, 1, 1, 3, 3, 4]])

    flood_fill(image, (0, 0), 5, inplace=True)

    expected = np.array([[5, 5, 5, 5, 5, 5, 5],
                        [5, 1, 1, 5, 2, 2, 5],
                        [5, 1, 1, 5, 2, 2, 5],
                        [1, 5, 5, 5, 5, 5, 3],
                        [5, 1, 1, 1, 3, 3, 4]])

    np.testing.assert_allclose(image, expected)


def test_1d():
    image = np.arange(11)
    expected = np.array([0, 1, -20, -20, -20, -20, -20, -20, -20, 9, 10])

    output = flood_fill(image, 5, -20, tolerance=3)
    output2 = flood_fill(image, (5,), -20, tolerance=3)

    np.testing.assert_equal(output, expected)
    np.testing.assert_equal(output, output2)


if __name__ == "__main__":
    np.testing.run_module_suite()
