import numpy as np
from numpy.testing import assert_array_equal
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
    result = clear_border(image.copy(), 1, 2)
    assert_array_equal(result, 2 * np.ones_like(image))


if __name__ == "__main__":
    np.testing.run_module_suite()
