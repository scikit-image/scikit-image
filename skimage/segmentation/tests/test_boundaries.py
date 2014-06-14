import numpy as np
from numpy.testing import assert_array_equal
from skimage.segmentation import find_boundaries, mark_boundaries


def test_find_boundaries():
    image = np.zeros((10, 10))
    image[2:7, 2:7] = 1

    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    result = find_boundaries(image)
    assert_array_equal(result, ref)


def test_mark_boundaries():
    image = np.zeros((10, 10))
    label_image = np.zeros((10, 10))
    label_image[2:7, 2:7] = 1

    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    result = mark_boundaries(image, label_image, color=(1, 1, 1)).mean(axis=2)
    assert_array_equal(result, ref)

    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 2, 0],
                    [0, 0, 1, 2, 2, 2, 2, 1, 2, 0],
                    [0, 0, 1, 2, 0, 0, 0, 1, 2, 0],
                    [0, 0, 1, 2, 0, 0, 0, 1, 2, 0],
                    [0, 0, 1, 2, 0, 0, 0, 1, 2, 0],
                    [0, 0, 1, 1, 1, 1, 1, 2, 2, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    result = mark_boundaries(image, label_image, color=(1, 1, 1),
                             outline_color=(2, 2, 2)).mean(axis=2)
    assert_array_equal(result, ref)


if __name__ == "__main__":
    np.testing.run_module_suite()
