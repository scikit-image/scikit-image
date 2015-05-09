import numpy as np
from numpy.testing import assert_array_equal, assert_
from skimage.segmentation import clear_border


def test_clear_border():
    image = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # test default case
    result = clear_border(image.copy())
    ref = image.copy()
    ref[2, 0] = 0
    ref[0, -2] = 0
    assert_array_equal(result, ref)

    # test buffer
    result = clear_border(image.copy(), 1)
    assert_array_equal(result, np.zeros(result.shape))

    # test background value
    result = clear_border(image.copy(), buffer_size=1, bgval=2)
    assert_array_equal(result, 2 * np.ones_like(image))


def test_clear_border_non_binary():
    image = np.array([[1, 2, 3, 1, 2],
                      [3, 3, 5, 4, 2],
                      [3, 4, 5, 4, 2],
                      [3, 3, 2, 1, 2]])

    result = clear_border(image)
    expected = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 5, 4, 0],
                         [0, 4, 5, 4, 0],
                         [0, 0, 0, 0, 0]])

    assert_array_equal(result, expected)
    assert_(not np.all(image == result))


def test_clear_border_non_binary_inplace():
    image = np.array([[1, 2, 3, 1, 2],
                      [3, 3, 5, 4, 2],
                      [3, 4, 5, 4, 2],
                      [3, 3, 2, 1, 2]])

    result = clear_border(image, in_place=True)
    expected = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 5, 4, 0],
                         [0, 4, 5, 4, 0],
                         [0, 0, 0, 0, 0]])

    assert_array_equal(result, expected)
    assert_array_equal(image, result)

if __name__ == "__main__":
    np.testing.run_module_suite()
